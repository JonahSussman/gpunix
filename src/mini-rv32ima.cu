// Copyright 2022 Charles Lohr, you may use this file or any portions herein under any of the BSD, MIT, or CC0 licenses.

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "default64mbdtc.h"

// Just default RAM amount is 64MB.
__constant__ uint32_t ram_amt;
uint32_t ram_amt_h = 64*1024*1024; // 64MB

__constant__ int fail_on_all_faults;
int fail_on_all_faults_h = 0;

struct MiniRV32IMAState;

static int64_t SimpleReadNumberInt( const char * number, int64_t defaultNumber );
static uint64_t GetTimeMicroseconds();
static void ResetKeyboardInput();
static void CaptureKeyboardInput();
__device__ uint32_t HandleException( uint32_t ir, uint32_t retval );
__device__ uint32_t HandleControlStore( MiniRV32IMAState* core, uint32_t addy, uint32_t val );
__device__ uint32_t HandleControlLoad( MiniRV32IMAState *core, uint32_t addy );
__device__ void HandleOtherCSRWrite( uint8_t * image, uint16_t csrno, uint32_t value );
__device__ int32_t HandleOtherCSRRead( uint8_t * image, uint16_t csrno );
static void MiniSleep();

__constant__ int is_kb_hit;
int is_kb_hit_h;
__device__ int IsKBHit();
static int IsKBHitHost();

__constant__ int kb_byte;
int kb_byte_h;
__device__ int ReadKBByte();
static int ReadKBByteHost();

// This is the functionality we want to override in the emulator.
//  think of this as the way the emulator's processor is connected to the outside world.
#define MINIRV32WARN( x... ) printf( x );
// #define MINIRV32_DECORATE  static
#define MINIRV32_DECORATE  __device__
#define MINI_RV32_RAM_SIZE ram_amt
#define MINIRV32_IMPLEMENTATION
#define MINIRV32_POSTEXEC( pc, ir, retval ) { if( retval > 0 ) { if( fail_on_all_faults ) { printf( "FAULT\n" ); return 3; } else retval = HandleException( ir, retval ); } }
#define MINIRV32_HANDLE_MEM_STORE_CONTROL( state, addy, val ) if( HandleControlStore( state, addy, val ) ) return val;
#define MINIRV32_HANDLE_MEM_LOAD_CONTROL( state, addy, rval ) rval = HandleControlLoad( state, addy );
#define MINIRV32_OTHERCSR_WRITE( csrno, value ) HandleOtherCSRWrite( image, csrno, value );
#define MINIRV32_OTHERCSR_READ( csrno, value ) value = HandleOtherCSRRead( image, csrno );
// ^ good v bad

#include "mini-rv32ima.h"

uint8_t * ram_image = 0;
struct MiniRV32IMAState * core;
const char * kernel_command_line = 0;

static void DumpState( struct MiniRV32IMAState * core, uint8_t * ram_image );

__global__ void MiniRV32IMAStepKernel(
	uint32_t* ret, struct MiniRV32IMAState * state, uint8_t * image, uint32_t vProcAddress, uint32_t elapsedUs, int count
) {
	*ret = MiniRV32IMAStep(
		state, image, vProcAddress, elapsedUs, count
	);
}

// NOTE(JonahSussman): Some things need to be simulated on the host, for example
// keyboard input stuff. This is extremely ugly. Sorry.
int32_t HandleHostCode( struct MiniRV32IMAState * state, uint8_t * image, uint32_t vProcAddress, uint32_t elapsedUs, int count ) {
	uint32_t new_timer = CSR( timerl ) + elapsedUs;
	if( new_timer < CSR( timerl ) ) CSR( timerh )++;
	CSR( timerl ) = new_timer;

	// Handle Timer interrupt.
	if( ( CSR( timerh ) > CSR( timermatchh ) || ( CSR( timerh ) == CSR( timermatchh ) && CSR( timerl ) > CSR( timermatchl ) ) ) && ( CSR( timermatchh ) || CSR( timermatchl ) ) )
	{
		CSR( extraflags ) &= ~4; // Clear WFI
		CSR( mip ) |= 1<<7; //MTIP of MIP // https://stackoverflow.com/a/61916199/2926815  Fire interrupt.
	}
	else
		CSR( mip ) &= ~(1<<7);

	// If WFI, don't run processor.
	if( CSR( extraflags ) & 4 )
		return 1;

	uint32_t trap = 0;
	uint32_t rval = 0;
	uint32_t pc = CSR( pc );
	uint32_t cycle = CSR( cyclel );

	if( ( CSR( mip ) & (1<<7) ) && ( CSR( mie ) & (1<<7) /*mtie*/ ) && ( CSR( mstatus ) & 0x8 /*mie*/) )
	{
		// Timer interrupt.
		trap = 0x80000007;
		pc -= 4;
	}
	else // No timer interrupt?  Execute a bunch of instructions.
	for( int icount = 0; icount < count; icount++ )
	{
		uint32_t ir = 0;
		rval = 0;
		cycle++;
		uint32_t ofs_pc = pc - MINIRV32_RAM_IMAGE_OFFSET;

		if( ofs_pc >= ram_amt_h )
		{
			trap = 1 + 1;  // Handle access violation on instruction read.
			break;
		}
		else if( ofs_pc & 3 )
		{
			trap = 1 + 0;  //Handle PC-misaligned access
			break;
		}
		else
		{
			ir = MINIRV32_LOAD4( ofs_pc );
			uint32_t rdid = (ir >> 7) & 0x1f;

			switch( ir & 0x7f )
			{
				case 0x03: // Load (0b0000011)
				{
					uint32_t rs1 = REG((ir >> 15) & 0x1f);
					uint32_t imm = ir >> 20;
					int32_t imm_se = imm | (( imm & 0x800 )?0xfffff000:0);
					uint32_t rsval = rs1 + imm_se;

					rsval -= MINIRV32_RAM_IMAGE_OFFSET;
					if( rsval >= ram_amt_h-3 )
					{
						rsval += MINIRV32_RAM_IMAGE_OFFSET;
						if( MINIRV32_MMIO_RANGE( rsval ) )  // UART, CLNT
						{
							// handle mmio

							auto addy = rsval;
							if( addy == 0x10000005 ) {
								is_kb_hit_h = IsKBHitHost();
								cudaMemcpyToSymbol( is_kb_hit, &is_kb_hit_h, sizeof(is_kb_hit_h) );
							}
							else if( addy == 0x10000000 ) {
								is_kb_hit_h = IsKBHitHost();
								cudaMemcpyToSymbol( is_kb_hit, &is_kb_hit_h, sizeof(is_kb_hit_h) );

								if (is_kb_hit_h) {
									kb_byte_h = ReadKBByteHost();
									cudaMemcpyToSymbol( kb_byte, &kb_byte_h, sizeof(kb_byte_h) );
								}
							}
						}
					}
				}
				case 0x73: // Zifencei+Zicsr  (0b1110011)
				{
					uint32_t csrno = ir >> 20;
					uint32_t microop = ( ir >> 12 ) & 0x7;
					if( (microop & 3) ) // It's a Zicsr function.
					{
						int rs1imm = (ir >> 15) & 0x1f;
						uint32_t rs1 = REG(rs1imm);
						uint32_t writeval = rs1;

						// https://raw.githubusercontent.com/riscv/virtual-memory/main/specs/663-Svpbmt.pdf
						// Generally, support for Zicsr
						switch( csrno )
						{
						case 0x340: rval = CSR( mscratch ); break;
						case 0x305: rval = CSR( mtvec ); break;
						case 0x304: rval = CSR( mie ); break;
						case 0xC00: rval = cycle; break;
						case 0x344: rval = CSR( mip ); break;
						case 0x341: rval = CSR( mepc ); break;
						case 0x300: rval = CSR( mstatus ); break; //mstatus
						case 0x342: rval = CSR( mcause ); break;
						case 0x343: rval = CSR( mtval ); break;
						case 0xf11: rval = 0xff0ff0ff; break; //mvendorid
						case 0x301: rval = 0x40401101; break; //misa (XLEN=32, IMA+X)
						//case 0x3B0: rval = 0; break; //pmpaddr0
						//case 0x3a0: rval = 0; break; //pmpcfg0
						//case 0xf12: rval = 0x00000000; break; //marchid
						//case 0xf13: rval = 0x00000000; break; //mimpid
						//case 0xf14: rval = 0x00000000; break; //mhartid
						default:
							// MINIRV32_OTHERCSR_READ( csrno, rval );
							if( csrno == 0x140 )
							{
								is_kb_hit_h = IsKBHitHost();
								cudaMemcpyToSymbol( is_kb_hit, &is_kb_hit_h, sizeof(is_kb_hit_h) );
								if( is_kb_hit_h )
								{
									kb_byte_h = ReadKBByteHost();
									cudaMemcpyToSymbol( kb_byte, &kb_byte_h, sizeof(kb_byte_h) );
								}
							}
							break;
						}
					}
				}
			}
		}
	}

	return 0;
}

int main( int argc, char ** argv )
{	
	cudaMemcpyToSymbol( ram_amt, &ram_amt_h, sizeof(ram_amt_h) );
	cudaMemcpyToSymbol( fail_on_all_faults, &fail_on_all_faults_h, sizeof(fail_on_all_faults_h) );

	int i;
	long long instct = -1;
	int show_help = 0;
	int time_divisor = 1;
	int fixed_update = 0;
	int do_sleep = 1;
	int single_step = 0;
	int dtb_ptr = 0;
	const char * image_file_name = 0;
	const char * dtb_file_name = 0;

	for( i = 1; i < argc; i++ )
	{
		const char * param = argv[i];
		int param_continue = 0; // Can combine parameters, like -lpt x
		do
		{
			if( param[0] == '-' || param_continue )
			{
				switch( param[1] )
				{
				case 'm': 
					if( ++i < argc ) {
						ram_amt_h = SimpleReadNumberInt( argv[i], ram_amt_h ); 
						cudaMemcpyToSymbol( ram_amt, &ram_amt_h, sizeof(ram_amt_h) );
						break;
					}
				case 'c': if( ++i < argc ) instct = SimpleReadNumberInt( argv[i], -1 ); break;
				case 'k': if( ++i < argc ) kernel_command_line = argv[i]; break;
				case 'f': image_file_name = (++i<argc)?argv[i]:0; break;
				case 'b': dtb_file_name = (++i<argc)?argv[i]:0; break;
				case 'l': param_continue = 1; fixed_update = 1; break;
				case 'p': param_continue = 1; do_sleep = 0; break;
				case 's': param_continue = 1; single_step = 1; break;
				case 'd': {
					param_continue = 1; 
					fail_on_all_faults_h = 1; 
					cudaMemcpyToSymbol( fail_on_all_faults, &fail_on_all_faults_h, sizeof(fail_on_all_faults_h) );
					break;
				}

				case 't': if( ++i < argc ) time_divisor = SimpleReadNumberInt( argv[i], 1 ); break;
				default:
					if( param_continue )
						param_continue = 0;
					else
						show_help = 1;
					break;
				}
			}
			else
			{
				show_help = 1;
				break;
			}
			param++;
		} while( param_continue );
	}
	if( show_help || image_file_name == 0 || time_divisor <= 0 )
	{
		fprintf( stderr, "./mini-rv32imaf [parameters]\n\t-m [ram amount]\n\t-f [running image]\n\t-k [kernel command line]\n\t-b [dtb file, or 'disable']\n\t-c instruction count\n\t-s single step with full processor state\n\t-t time divion base\n\t-l lock time base to instruction count\n\t-p disable sleep when wfi\n\t-d fail out immediately on all faults\n" );
		return 1;
	}

	cudaMallocManaged((void**)&ram_image, ram_amt_h);
	if( !ram_image )
	{
		fprintf( stderr, "Error: could not allocate system image.\n" );
		return -4;
	}

restart:
	{
		FILE * f = fopen( image_file_name, "rb" );
		if( !f || ferror( f ) )
		{
			fprintf( stderr, "Error: \"%s\" not found\n", image_file_name );
			return -5;
		}
		fseek( f, 0, SEEK_END );
		long flen = ftell( f );
		fseek( f, 0, SEEK_SET );
		if( flen > ram_amt_h )
		{
			fprintf( stderr, "Error: Could not fit RAM image (%ld bytes) into %d\n", flen, ram_amt_h );
			return -6;
		}

		memset( ram_image, 0, ram_amt_h );
		if( fread( ram_image, flen, 1, f ) != 1)
		{
			fprintf( stderr, "Error: Could not load image.\n" );
			return -7;
		}
		fclose( f );

		if( dtb_file_name )
		{
			if( strcmp( dtb_file_name, "disable" ) == 0 )
			{
				// No DTB reading.
			}
			else
			{
				f = fopen( dtb_file_name, "rb" );
				if( !f || ferror( f ) )
				{
					fprintf( stderr, "Error: \"%s\" not found\n", dtb_file_name );
					return -5;
				}
				fseek( f, 0, SEEK_END );
				long dtblen = ftell( f );
				fseek( f, 0, SEEK_SET );
				dtb_ptr = ram_amt_h - dtblen - sizeof( struct MiniRV32IMAState );
				if( fread( ram_image + dtb_ptr, dtblen, 1, f ) != 1 )
				{
					fprintf( stderr, "Error: Could not open dtb \"%s\"\n", dtb_file_name );
					return -9;
				}
				fclose( f );
			}
		}
		else
		{
			// Load a default dtb.
			dtb_ptr = ram_amt_h - sizeof(default64mbdtb) - sizeof( struct MiniRV32IMAState );
			memcpy( ram_image + dtb_ptr, default64mbdtb, sizeof( default64mbdtb ) );
			if( kernel_command_line )
			{
				strncpy( (char*)( ram_image + dtb_ptr + 0xc0 ), kernel_command_line, 54 );
			}
		}
	}

	CaptureKeyboardInput();

	// The core lives at the end of RAM.
	core = (struct MiniRV32IMAState *)(ram_image + ram_amt_h - sizeof( struct MiniRV32IMAState ));
	core->pc = MINIRV32_RAM_IMAGE_OFFSET;
	core->regs[10] = 0x00; //hart ID
	core->regs[11] = dtb_ptr?(dtb_ptr+MINIRV32_RAM_IMAGE_OFFSET):0; //dtb_pa (Must be valid pointer) (Should be pointer to dtb)
	core->extraflags |= 3; // Machine-mode.

	if( dtb_file_name == 0 )
	{
		// Update system ram size in DTB (but if and only if we're using the default DTB)
		// Warning - this will need to be updated if the skeleton DTB is ever modified.
		uint32_t * dtb = (uint32_t*)(ram_image + dtb_ptr);
		if( dtb[0x13c/4] == 0x00c0ff03 )
		{
			uint32_t validram = dtb_ptr;
			dtb[0x13c/4] = (validram>>24) | ((( validram >> 16 ) & 0xff) << 8 ) | (((validram>>8) & 0xff ) << 16 ) | ( ( validram & 0xff) << 24 );
		}
	}

	// Image is loaded.
	uint64_t rt;
	uint64_t lastTime = (fixed_update)?0:(GetTimeMicroseconds()/time_divisor);
	int instrs_per_flip = single_step?1:1024;
	for( rt = 0; rt < instct+1 || instct < 0; rt += instrs_per_flip )
	{
		uint64_t * this_ccount = ((uint64_t*)&core->cyclel);
		uint32_t elapsedUs = 0;
		if( fixed_update )
			elapsedUs = *this_ccount / time_divisor - lastTime;
		else
			elapsedUs = GetTimeMicroseconds()/time_divisor - lastTime;
		lastTime += elapsedUs;

		// if( single_step )
		// 	DumpState( core, ram_image);

		uint32_t* ret;
		cudaMallocManaged((void**)&ret, sizeof(int));

		// Set things up for the next processor step.
		
		// TODO(JonahSussman): Only do the keyboard check if needed.
		// is_kb_hit_h = IsKBHitHost();
		// cudaMemcpyToSymbol( is_kb_hit, &is_kb_hit_h, sizeof(is_kb_hit_h) );
		
		MiniRV32IMAState* state = (MiniRV32IMAState*)malloc(sizeof(MiniRV32IMAState));
		for (int i = 0; i < sizeof(MiniRV32IMAState); ++i) {
			((uint8_t*)state)[i] = ((uint8_t*)core)[i];
		}
		HandleHostCode(state, ram_image, 0, elapsedUs, instrs_per_flip);
		free(state);

		MiniRV32IMAStepKernel<<<1, 1>>>(
			ret, core, ram_image, 0, elapsedUs, instrs_per_flip
		); // Execute up to 1024 cycles before breaking out.
		cudaDeviceSynchronize();

		switch( *ret )
		{
			case 0: break;
			case 1: if( do_sleep ) MiniSleep(); *this_ccount += instrs_per_flip; break;
			case 3: instct = 0; break;
			case 0x7777: goto restart;	//syscon code for restart
			case 0x5555: printf( "POWEROFF@0x%08x%08x\n", core->cycleh, core->cyclel ); return 0; //syscon code for power-off
			default: printf( "Unknown failure\n" ); break;
		}
	}

	DumpState( core, ram_image);
}


//////////////////////////////////////////////////////////////////////////
// Platform-specific functionality
//////////////////////////////////////////////////////////////////////////


#if defined(WINDOWS) || defined(WIN32) || defined(_WIN32)

#include <windows.h>
#include <conio.h>

#define strtoll _strtoi64

static void CaptureKeyboardInput()
{
	system(""); // Poorly documented tick: Enable VT100 Windows mode.
}

static void ResetKeyboardInput()
{
}

static void MiniSleep()
{
	Sleep(1);
}

static uint64_t GetTimeMicroseconds()
{
	static LARGE_INTEGER lpf;
	LARGE_INTEGER li;

	if( !lpf.QuadPart )
		QueryPerformanceFrequency( &lpf );

	QueryPerformanceCounter( &li );
	return ((uint64_t)li.QuadPart * 1000000LL) / (uint64_t)lpf.QuadPart;
}


static int IsKBHit()
{
	return _kbhit();
}

static int ReadKBByte()
{
	// This code is kind of tricky, but used to convert windows arrow keys
	// to VT100 arrow keys.
	static int is_escape_sequence = 0;
	int r;
	if( is_escape_sequence == 1 )
	{
		is_escape_sequence++;
		return '[';
	}

	r = _getch();

	if( is_escape_sequence )
	{
		is_escape_sequence = 0;
		switch( r )
		{
			case 'H': return 'A'; // Up
			case 'P': return 'B'; // Down
			case 'K': return 'D'; // Left
			case 'M': return 'C'; // Right
			case 'G': return 'H'; // Home
			case 'O': return 'F'; // End
			default: return r; // Unknown code.
		}
	}
	else
	{
		switch( r )
		{
			case 13: return 10; //cr->lf
			case 224: is_escape_sequence = 1; return 27; // Escape arrow keys
			default: return r;
		}
	}
}

#else

#include <sys/ioctl.h>
#include <termios.h>
#include <unistd.h>
#include <signal.h>
#include <sys/time.h>

// TODO(JonahSussman): Figure out signum
static void CtrlC(int signum)
{
	DumpState( core, ram_image);
	exit( 0 );
}

// Override keyboard, so we can capture all keyboard input for the VM.
static void CaptureKeyboardInput()
{
	// Hook exit, because we want to re-enable keyboard.
	atexit(ResetKeyboardInput);
	signal(SIGINT, CtrlC);

	struct termios term;
	tcgetattr(0, &term);
	term.c_lflag &= ~(ICANON | ECHO); // Disable echo as well
	tcsetattr(0, TCSANOW, &term);
}

static void ResetKeyboardInput()
{
	// Re-enable echo, etc. on keyboard.
	struct termios term;
	tcgetattr(0, &term);
	term.c_lflag |= ICANON | ECHO;
	tcsetattr(0, TCSANOW, &term);
}

static void MiniSleep()
{
	usleep(500);
}

static uint64_t GetTimeMicroseconds()
{
	struct timeval tv;
	gettimeofday( &tv, 0 );
	return tv.tv_usec + ((uint64_t)(tv.tv_sec)) * 1000000LL;
}

static int is_eofd;

__device__ int ReadKBByte() {
	return kb_byte;
}

static int ReadKBByteHost()
{
	if( is_eofd ) return 0xffffffff;
	char rxchar = 0;
	int rread = read(fileno(stdin), (char*)&rxchar, 1);

	if( rread > 0 ) // Tricky: getchar can't be used with arrow keys.
		return rxchar;
	else
		return -1;
}

__device__ int IsKBHit() {
	return is_kb_hit;
}

static int IsKBHitHost()
{
	if( is_eofd ) return -1;
	int byteswaiting;
	ioctl(0, FIONREAD, &byteswaiting);
	if( !byteswaiting && write( fileno(stdin), 0, 0 ) != 0 ) { is_eofd = 1; return -1; } // Is end-of-file for 
	return !!byteswaiting;
}


#endif


//////////////////////////////////////////////////////////////////////////
// Rest of functions functionality
//////////////////////////////////////////////////////////////////////////

__device__ uint32_t HandleException( uint32_t ir, uint32_t code )
{
	// Weird opcode emitted by duktape on exit.
	if( code == 3 )
	{
		// Could handle other opcodes here.
	}
	return code;
}

__device__ uint32_t HandleControlStore( MiniRV32IMAState* core, uint32_t addy, uint32_t val )
{
	if( addy == 0x10000000 ) //UART 8250 / 16550 Data Buffer
	{
		printf( "%c", val );
		// fflush( stdout );
	}
	else if( addy == 0x11004004 ) //CLNT
		core->timermatchh = val;
	else if( addy == 0x11004000 ) //CLNT
		core->timermatchl = val;
	else if( addy == 0x11100000 ) //SYSCON (reboot, poweroff, etc.)
	{
		core->pc = core->pc + 4;
		return val; // NOTE: PC will be PC of Syscon.
	}
	return 0;
}


__device__ uint32_t HandleControlLoad( MiniRV32IMAState *core, uint32_t addy )
{
	// Emulating a 8250 / 16550 UART
	if( addy == 0x10000005 )
		return 0x60 | IsKBHit();
	else if( addy == 0x10000000 && IsKBHit() )
		return ReadKBByte();
	else if( addy == 0x1100bffc ) // https://chromitem-soc.readthedocs.io/en/latest/clint.html
		return core->timerh;
	else if( addy == 0x1100bff8 )
		return core->timerl;
	return 0;
}

__device__ void HandleOtherCSRWrite( uint8_t * image, uint16_t csrno, uint32_t value )
{
	if( csrno == 0x136 )
	{
		printf( "%d", value ); 
		// fflush( stdout );
	}
	if( csrno == 0x137 )
	{
		printf( "%08x", value ); 
		// fflush( stdout );
	}
	else if( csrno == 0x138 )
	{
		//Print "string"
		uint32_t ptrstart = value - MINIRV32_RAM_IMAGE_OFFSET;
		uint32_t ptrend = ptrstart;
		if( ptrstart >= MINI_RV32_RAM_SIZE )
			printf( "DEBUG PASSED INVALID PTR (%08x)\n", value );
		while( ptrend < MINI_RV32_RAM_SIZE )
		{
			if( image[ptrend] == 0 ) break;
			ptrend++;
		}
		if( ptrend != ptrstart )
			// fwrite( image + ptrstart, ptrend - ptrstart, 1, stdout );
			printf("%.*s", (int)(ptrend - ptrstart), (char*)(image + ptrstart));
	}
	else if( csrno == 0x139 )
	{
		// putchar( value ); 
		printf( "%c", value );
		// fflush( stdout );
	}
}

__device__ int32_t HandleOtherCSRRead( uint8_t * image, uint16_t csrno )
{
	if( csrno == 0x140 )
	{
		if( !IsKBHit() ) return -1;
		return ReadKBByte();
	}
	return 0;
}

static int64_t SimpleReadNumberInt( const char * number, int64_t defaultNumber )
{
	if( !number || !number[0] ) return defaultNumber;
	int radix = 10;
	if( number[0] == '0' )
	{
		char nc = number[1];
		number+=2;
		if( nc == 0 ) return 0;
		else if( nc == 'x' ) radix = 16;
		else if( nc == 'b' ) radix = 2;
		else { number--; radix = 8; }
	}
	char * endptr;
	uint64_t ret = strtoll( number, &endptr, radix );
	if( endptr == number )
	{
		return defaultNumber;
	}
	else
	{
		return ret;
	}
}

static void DumpState( struct MiniRV32IMAState * core, uint8_t * ram_image )
{
	uint32_t pc = core->pc;
	uint32_t pc_offset = pc - MINIRV32_RAM_IMAGE_OFFSET;
	uint32_t ir = 0;

	printf( "PC: %08x ", pc );
	if( pc_offset >= 0 && pc_offset < ram_amt_h - 3 )
	{
		ir = *((uint32_t*)(&((uint8_t*)ram_image)[pc_offset]));
		printf( "[0x%08x] ", ir ); 
	}
	else
		printf( "[xxxxxxxxxx] " ); 
	uint32_t * regs = core->regs;
	printf( "Z:%08x ra:%08x sp:%08x gp:%08x tp:%08x t0:%08x t1:%08x t2:%08x s0:%08x s1:%08x a0:%08x a1:%08x a2:%08x a3:%08x a4:%08x a5:%08x ",
		regs[0], regs[1], regs[2], regs[3], regs[4], regs[5], regs[6], regs[7],
		regs[8], regs[9], regs[10], regs[11], regs[12], regs[13], regs[14], regs[15] );
	printf( "a6:%08x a7:%08x s2:%08x s3:%08x s4:%08x s5:%08x s6:%08x s7:%08x s8:%08x s9:%08x s10:%08x s11:%08x t3:%08x t4:%08x t5:%08x t6:%08x\n",
		regs[16], regs[17], regs[18], regs[19], regs[20], regs[21], regs[22], regs[23],
		regs[24], regs[25], regs[26], regs[27], regs[28], regs[29], regs[30], regs[31] );
}

