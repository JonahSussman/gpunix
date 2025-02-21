#include <stdint.h>

struct VMConfigLoadState {
  
};

struct VirtMachineParams {
  char *cfg_filename;
  // const VirtMachineClass *vmc;
  char *machine_name;
  uint64_t ram_size;
  bool rtc_real_time;
  bool rtc_local_time;
  char *display_device; /* NULL means no display */
  int width, height; /* graphic width & height */
  // CharacterDevice *console;
  // VMDriveEntry tab_drive[MAX_DRIVE_DEVICE];
  int drive_count;
  // VMFSEntry tab_fs[MAX_FS_DEVICE];
  int fs_count;
  // VMEthEntry tab_eth[MAX_ETH_DEVICE];
  int eth_count;

  char *cmdline; /* bios or kernel command line */
  bool accel_enable; /* enable acceleration (KVM) */
  char *input_device; /* NULL means no input */
  
  /* kernel, bios and other auxiliary files */
  // VMFileEntry files[VM_FILE_COUNT];

  __host__ VirtMachineParams() {
    memset(this, 0, sizeof(VirtMachineParams));
  }

  __host__ void load_config_file(const char *filename) {

  }
};

struct VirtMachine {
  // // Network
  // EthernetDevice *net;

  // // Console
  // VIRTIODevice *console_dev;
  // CharacterDevice *console;
  
  // // Graphics
  // FBDevice *fb_dev;

  // // Methods


};

