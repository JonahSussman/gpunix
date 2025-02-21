# if set, network filesystem is enabled. libcurl and libcrypto
# (openssl) must be installed.
# CONFIG_FS_NET=y
# SDL support (optional)
# CONFIG_SDL=y
# if set, compile the 128 bit emulator. Note: the 128 bit target does
# not compile if gcc does not support the int128 type (32 bit hosts).
# CONFIG_INT128=y
# win32 build (not usable yet)
# CONFIG_WIN32=y
# user space network redirector
# CONFIG_SLIRP=y

CC = nvcc

CFLAGS =  -O2 -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -MMD -arch=sm_61
CFLAGS += -Wno-deprecated-gpu-targets -diag-suppress 2464
CFLAGS += -D_GNU_SOURCE -DCONFIG_VERSION=\"$(shell cat VERSION)\"

LDFLAGS = -Wno-deprecated-gpu-targets

PROGS += riscv-emu-cuda

ifdef CONFIG_FS_NET
PROGS += build_filelist splitimg
endif


all: $(PROGS)


EMU_OBJS := main.o

# EMU_OBJS:=virtio.o pci.o fs.o cutils.o iomem.o simplefb.o json.o machine.o temu.o

# ifdef CONFIG_SLIRP
# CFLAGS+=-DCONFIG_SLIRP
# EMU_OBJS+=$(addprefix slirp/, bootp.o ip_icmp.o mbuf.o slirp.o tcp_output.o cksum.o ip_input.o misc.o socket.o tcp_subr.o udp.o if.o ip_output.o sbuf.o tcp_input.o tcp_timer.o)
# endif

# EMU_OBJS+=fs_disk.o
EMU_LIBS=-lrt

ifdef CONFIG_FS_NET
CFLAGS+=-DCONFIG_FS_NET
# EMU_OBJS+=fs_net.o fs_wget.o fs_utils.o block_net.o
EMU_LIBS+=-lcurl -lcrypto
endif # CONFIG_FS_NET

ifdef CONFIG_SDL
EMU_LIBS+=-lSDL
# EMU_OBJS+=sdl.o
CFLAGS+=-DCONFIG_SDL
endif

# EMU_OBJS += riscv_machine.o softfp.o riscv_cpu32.o riscv_cpu64.o

ifdef CONFIG_INT128
CFLAGS+=-DCONFIG_RISCV_MAX_XLEN=128
EMU_OBJS+=riscv_cpu128.o
else
CFLAGS+=-DCONFIG_RISCV_MAX_XLEN=64
endif


riscv-emu-cuda: $(EMU_OBJS)
	$(CC) $(LDFLAGS) -o $@ $^ $(EMU_LIBS)

%.o: %.cu
	$(CC) $(CFLAGS) --device-c -o $@ $<

riscv_cpu32.o: riscv_cpu.cu
	$(CC) $(CFLAGS) -DMAX_XLEN=32 -c -o $@ $<

riscv_cpu64.o: riscv_cpu.cu
	$(CC) $(CFLAGS) -DMAX_XLEN=64 -c -o $@ $<

riscv_cpu128.o: riscv_cpu.cu
	$(CC) $(CFLAGS) -DMAX_XLEN=128 -c -o $@ $<


build_filelist: build_filelist.o fs_utils.o cutils.o
	$(CC) $(LDFLAGS) -o $@ $^ -lm

splitimg: splitimg.o
	$(CC) $(LDFLAGS)  -o $@ $^


clean:
	rm -f *.o *.d *~ $(PROGS) slirp/*.o slirp/*.d slirp/*~

-include $(wildcard *.d)
-include $(wildcard slirp/*.d)
