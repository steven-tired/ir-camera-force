/*
 * Copyright (C) 2015-2016 Thomas <tomas123 @ EEVblog Electronics Community Forum>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 */

#include <errno.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <libusb.h>
#include <unistd.h>
#include <time.h>
 
#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <limits.h>
#include <sys/stat.h>

#include "jpeglib.h"

#include "plank.h"

// -- define v4l2 ---------------
#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <string.h>
#include <fcntl.h>
#include <assert.h>

#define VIDEO_DEVICE0 "/dev/video1"  // gray scale thermal image
#define FRAME_WIDTH0  160
#define FRAME_HEIGHT0 120

#define VIDEO_DEVICE1 "/dev/video20" // color visible image
#define FRAME_WIDTH1  640
#define FRAME_HEIGHT1 480

#define VIDEO_DEVICE2 "/dev/video21" // colorized thermal image
#define FRAME_WIDTH2  160
#define FRAME_HEIGHT2 128

#define FRAME_FORMAT0 V4L2_PIX_FMT_GREY
#define FRAME_FORMAT1 V4L2_PIX_FMT_MJPEG
#define FRAME_FORMAT2 V4L2_PIX_FMT_RGB24

struct v4l2_capability vid_caps0;
struct v4l2_capability vid_caps1;
struct v4l2_capability vid_caps2;

struct v4l2_format vid_format0;
struct v4l2_format vid_format1;
struct v4l2_format vid_format2;

size_t framesize0;
size_t linewidth0;

size_t framesize1;
size_t linewidth1;

size_t framesize2;
size_t linewidth2;

     
const char *video_device0=VIDEO_DEVICE0;
const char *video_device1=VIDEO_DEVICE1;
const char *video_device2=VIDEO_DEVICE2;

int fdwr0 = 0;
int fdwr1 = 0;
int fdwr2 = 0;

// -- end define v4l2 ---------------

 #define VENDOR_ID 0x09cb
 #define PRODUCT_ID 0x1996

 static struct libusb_device_handle *devh = NULL;
int filecount=0;
struct timeval t1, t2;
long long fps_t;
int thermal_layout_reported = 0;

static const char *raw_output_dir = NULL;
static unsigned int raw_frame_limit = 0;
static unsigned int raw_frames_written = 0;
static unsigned long long thermal_frame_index = 0;
static unsigned long long previous_raw_hash = 0;
static int previous_raw_hash_valid = 0;
static int fixed_raw_range_enabled = 0;
static unsigned short fixed_raw_low = 0;
static unsigned short fixed_raw_high = 0;
static int legacy_generic_temperature_overlay = 0;
static int thermal_loopback_write_failed = 0;
 
 int FFC =   0; // detect FFC

static void write_thermal_loopback_frame(const unsigned char *frame, size_t frame_size)
{
  ssize_t written = write(fdwr2, frame, frame_size);

  if (written == (ssize_t)frame_size) {
    if (thermal_loopback_write_failed) {
      fprintf(stderr, "thermal RGB loopback write recovered\n");
      thermal_loopback_write_failed = 0;
    }
    return;
  }
  if (!thermal_loopback_write_failed) {
    if (written < 0)
      fprintf(stderr, "thermal RGB loopback write failed: %s\n", strerror(errno));
    else
      fprintf(stderr, "thermal RGB loopback write failed: wrote %zd of %zu bytes\n", written, frame_size);
    thermal_loopback_write_failed = 1;
  }
}

// -- buffer for EP 0x85 chunks ---------------
 #define BUF85SIZE 1048576  // size got from android app
 int buf85pointer = 0;
 unsigned char buf85[BUF85SIZE];
  
 void print_format(struct v4l2_format*vid_format) {
  printf("     vid_format->type                =%d\n",     vid_format->type );
  printf("     vid_format->fmt.pix.width       =%d\n",     vid_format->fmt.pix.width );
  printf("     vid_format->fmt.pix.height      =%d\n",     vid_format->fmt.pix.height );
  printf("     vid_format->fmt.pix.pixelformat =%d\n",     vid_format->fmt.pix.pixelformat);
  printf("     vid_format->fmt.pix.sizeimage   =%u\n",     vid_format->fmt.pix.sizeimage );
  printf("     vid_format->fmt.pix.field       =%d\n",     vid_format->fmt.pix.field );
  printf("     vid_format->fmt.pix.bytesperline=%d\n",     vid_format->fmt.pix.bytesperline );
  printf("     vid_format->fmt.pix.colorspace  =%d\n",     vid_format->fmt.pix.colorspace );
}

//#include "font.h" 
#include "font5x7.h" 
void font_write(unsigned char *fb, int x, int y, const char *string)
{
  int rx, ry;
  while (*string) {
    for (ry = 0; ry < 5; ++ry) {
      for (rx = 0; rx < 7; ++rx) {
        int v = (font5x7_basic[((*string) & 0x7F) - CHAR_OFFSET][ry] >> (rx)) & 1;
//	fb[(y+ry) * 160 + (x + rx)] = v ? 0 : 0xFF;                       // black / white
//	fb[(y+rx) * 160 + (x + ry)] = v ? 0 : 0xFF;                       // black / white

        fb[(y+rx) * 160 + (x + ry)] = v ? 0 : fb[(y+rx) * 160 + (x + ry)];  // transparent
      }
    }
    string++;
    x += 6;
  }
}

double raw2temperature(unsigned short RAW)
{
 // mystery correction factor
 RAW *=4;
 // calc amount of radiance of reflected objects ( Emissivity < 1 )
 double RAWrefl=PlanckR1/(PlanckR2*(exp(PlanckB/(TempReflected+273.15))-PlanckF))-PlanckO;
 // get displayed object temp max/min
 double RAWobj=(RAW-(1-Emissivity)*RAWrefl)/Emissivity;
 // calc object temperature
 return PlanckB/log(PlanckR1/(PlanckR2*(RAWobj+PlanckO))+PlanckF)-273.15;  
}

static unsigned short read_le16(const unsigned char *p)
{
  return (unsigned short)(p[0] | (p[1] << 8));
}

static void decode_g2_thermal(const unsigned char *thermal, unsigned short *pix)
{
  int x, y;

  for (y = 0; y < 120; ++y)
  {
    for (x = 0; x < 160; ++x) {
      if (x < 80)
        pix[y * 160 + x] = read_le16(&thermal[2 * (y * 164 + x) + 4]);
      else
        pix[y * 160 + x] = read_le16(&thermal[2 * (y * 164 + x) + 8]);
    }
  }
}

static void decode_lepton2_sensor(const unsigned char *thermal, unsigned short *sensor_pix)
{
  int x, y;

  for (y = 0; y < 60; ++y)
  {
    const unsigned char *row = thermal + y * 164 + 4;
    for (x = 0; x < 80; ++x) {
      sensor_pix[y * 80 + x] = read_le16(&row[x * 2]);
    }
  }
}

static void upscale_lepton2_sensor(const unsigned short *sensor_pix, unsigned short *pix)
{
  int x, y;

  for (y = 0; y < 60; ++y)
  {
    for (x = 0; x < 80; ++x) {
      unsigned short v = sensor_pix[y * 80 + x];
      int yy = y * 2;
      int xx = x * 2;

      pix[yy * 160 + xx] = v;
      pix[yy * 160 + xx + 1] = v;
      pix[(yy + 1) * 160 + xx] = v;
      pix[(yy + 1) * 160 + xx + 1] = v;
    }
  }
}

static int compare_u16(const void *left, const void *right)
{
  unsigned short a = *(const unsigned short *)left;
  unsigned short b = *(const unsigned short *)right;

  return (a > b) - (a < b);
}

static unsigned short median_u16(const unsigned short *values, size_t count)
{
  unsigned short sorted[160 * 120];

  if (count == 0 || count > sizeof(sorted) / sizeof(sorted[0]))
    return 0;
  memcpy(sorted, values, count * sizeof(sorted[0]));
  qsort(sorted, count, sizeof(sorted[0]), compare_u16);
  if (count % 2)
    return sorted[count / 2];
  return (unsigned short)((sorted[count / 2 - 1] + sorted[count / 2]) / 2);
}

static unsigned long long hash_u16(const unsigned short *values, size_t count)
{
  size_t index;
  unsigned long long hash = 1469598103934665603ULL;

  for (index = 0; index < count; ++index) {
    hash ^= values[index] & 0xff;
    hash *= 1099511628211ULL;
    hash ^= values[index] >> 8;
    hash *= 1099511628211ULL;
  }
  return hash;
}

static int write_u16le(FILE *stream, const unsigned short *values, size_t count)
{
  size_t index;

  for (index = 0; index < count; ++index) {
    if (fputc(values[index] & 0xff, stream) == EOF ||
        fputc(values[index] >> 8, stream) == EOF)
      return -1;
  }
  return 0;
}

static void write_raw_frame(const unsigned short *values, int width, int height,
                            uint32_t thermal_size, const char *source_byte_order,
                            const char *ffc_state)
{
  char raw_path[PATH_MAX];
  char metadata_path[PATH_MAX];
  char raw_name[64];
  char metadata_name[64];
  size_t count = (size_t)width * height;
  size_t index;
  unsigned short raw_min = USHRT_MAX;
  unsigned short raw_max = 0;
  unsigned short raw_median;
  unsigned long long hash;
  unsigned long long timestamp_ns;
  int repeated_frame;
  struct timespec timestamp;
  FILE *raw_file;
  FILE *metadata_file;

  if (raw_output_dir == NULL || width <= 0 || height <= 0)
    return;
  if (raw_frame_limit != 0 && raw_frames_written >= raw_frame_limit)
    return;

  for (index = 0; index < count; ++index) {
    if (values[index] < raw_min)
      raw_min = values[index];
    if (values[index] > raw_max)
      raw_max = values[index];
  }
  raw_median = median_u16(values, count);
  hash = hash_u16(values, count);
  repeated_frame = previous_raw_hash_valid && hash == previous_raw_hash;
  previous_raw_hash = hash;
  previous_raw_hash_valid = 1;
  clock_gettime(CLOCK_MONOTONIC, &timestamp);
  timestamp_ns = (unsigned long long)timestamp.tv_sec * 1000000000ULL + timestamp.tv_nsec;

  snprintf(raw_name, sizeof(raw_name), "raw_frame_%06llu.u16le", thermal_frame_index);
  snprintf(metadata_name, sizeof(metadata_name), "raw_frame_%06llu.json", thermal_frame_index);
  snprintf(raw_path, sizeof(raw_path), "%s/%s", raw_output_dir, raw_name);
  snprintf(metadata_path, sizeof(metadata_path), "%s/%s", raw_output_dir, metadata_name);

  raw_file = fopen(raw_path, "wb");
  if (raw_file == NULL) {
    fprintf(stderr, "raw capture write failed for %s: %s\n", raw_path, strerror(errno));
    return;
  }
  if (write_u16le(raw_file, values, count) != 0) {
    fclose(raw_file);
    fprintf(stderr, "raw capture write failed for %s: %s\n", raw_path, strerror(errno));
    return;
  }
  if (fclose(raw_file) != 0) {
    fprintf(stderr, "raw capture write failed for %s: %s\n", raw_path, strerror(errno));
    return;
  }

  metadata_file = fopen(metadata_path, "w");
  if (metadata_file == NULL) {
    fprintf(stderr, "raw capture metadata write failed for %s: %s\n", metadata_path, strerror(errno));
    return;
  }
  fprintf(metadata_file,
          "{\n"
          "  \"schema_version\": 1,\n"
          "  \"frame_index\": %llu,\n"
          "  \"monotonic_timestamp_ns\": %llu,\n"
          "  \"camera_timestamp_available\": false,\n"
          "  \"camera_timestamp\": null,\n"
          "  \"width\": %d,\n"
          "  \"height\": %d,\n"
          "  \"dtype\": \"uint16\",\n"
          "  \"byte_order\": \"little\",\n"
          "  \"source_payload_byte_order\": \"%s\",\n"
          "  \"thermal_payload_size\": %u,\n"
          "  \"raw_file\": \"%s\",\n"
          "  \"ffc_state\": \"%s\",\n"
          "  \"raw_min\": %u,\n"
          "  \"raw_median\": %u,\n"
          "  \"raw_max\": %u,\n"
          "  \"dropped_frame_flag\": false,\n"
          "  \"dropped_frame_observable\": false,\n"
          "  \"repeated_frame_flag\": %s,\n"
          "  \"display_mapping\": {\"mode\": \"%s\", \"raw_low\": %d, \"raw_high\": %d},\n"
          "  \"calibration\": {\"used\": false, \"source\": \"none; src/plank.h has generic FLIR ONE JPEG EXIF constants only\"}\n"
          "}\n",
          thermal_frame_index, timestamp_ns, width, height, source_byte_order,
          thermal_size, raw_name, ffc_state, raw_min, raw_median, raw_max,
          repeated_frame ? "true" : "false",
          fixed_raw_range_enabled ? "fixed_raw_counts" : "dynamic_per_frame_min_max",
          fixed_raw_range_enabled ? fixed_raw_low : -1,
          fixed_raw_range_enabled ? fixed_raw_high : -1);
  if (fclose(metadata_file) != 0)
    fprintf(stderr, "raw capture metadata close failed for %s: %s\n", metadata_path, strerror(errno));
  else {
    ++raw_frames_written;
    ++thermal_frame_index;
  }
}


void startv4l2()
{
     int ret_code = 0;

     int i;
     int k=1;
/*     
//open video_device0
     printf("using output device: %s\n", video_device0);
     
     fdwr0 = open(video_device0, O_RDWR);
     assert(fdwr0 >= 0);

     ret_code = ioctl(fdwr0, VIDIOC_QUERYCAP, &vid_caps0);
     assert(ret_code != -1);

     memset(&vid_format0, 0, sizeof(vid_format0));

     ret_code = ioctl(fdwr0, VIDIOC_G_FMT, &vid_format0);

     linewidth0=FRAME_WIDTH0;
     framesize0=FRAME_WIDTH0*FRAME_HEIGHT0*1; // 8 Bit

     vid_format0.type = V4L2_BUF_TYPE_VIDEO_OUTPUT;
     vid_format0.fmt.pix.width = FRAME_WIDTH0;
     vid_format0.fmt.pix.height = FRAME_HEIGHT0;
     vid_format0.fmt.pix.pixelformat = FRAME_FORMAT0;
     vid_format0.fmt.pix.sizeimage = framesize0;
     vid_format0.fmt.pix.field = V4L2_FIELD_NONE;
     vid_format0.fmt.pix.bytesperline = linewidth0;
     vid_format0.fmt.pix.colorspace = V4L2_COLORSPACE_SRGB;

     // set data format
     ret_code = ioctl(fdwr0, VIDIOC_S_FMT, &vid_format0);
     assert(ret_code != -1);

     print_format(&vid_format0);
*/     
//open video_device1
     printf("using output device: %s\n", video_device1);
     
     fdwr1 = open(video_device1, O_RDWR);
     assert(fdwr1 >= 0);

     ret_code = ioctl(fdwr1, VIDIOC_QUERYCAP, &vid_caps1);
     assert(ret_code != -1);

     memset(&vid_format1, 0, sizeof(vid_format1));

     ret_code = ioctl(fdwr1, VIDIOC_G_FMT, &vid_format1);

     linewidth1=FRAME_WIDTH1;
     framesize1=FRAME_WIDTH1*FRAME_HEIGHT1*1; // 8 Bit ??

     vid_format1.type = V4L2_BUF_TYPE_VIDEO_OUTPUT;
     vid_format1.fmt.pix.width = FRAME_WIDTH1;
     vid_format1.fmt.pix.height = FRAME_HEIGHT1;
     vid_format1.fmt.pix.pixelformat = FRAME_FORMAT1;
     vid_format1.fmt.pix.sizeimage = framesize1;
     vid_format1.fmt.pix.field = V4L2_FIELD_NONE;
     vid_format1.fmt.pix.bytesperline = linewidth1;
     vid_format1.fmt.pix.colorspace = V4L2_COLORSPACE_SRGB;

     // set data format
     ret_code = ioctl(fdwr1, VIDIOC_S_FMT, &vid_format1);
     if (ret_code == -1) {
       fprintf(stderr,
               "VIDIOC_S_FMT %s failed: errno=%d (%s), format=%ux%u %.4s bytesperline=%u sizeimage=%u\\n",
               video_device1,
               errno,
               strerror(errno),
               vid_format1.fmt.pix.width,
               vid_format1.fmt.pix.height,
               (char *)&vid_format1.fmt.pix.pixelformat,
               vid_format1.fmt.pix.bytesperline,
               vid_format1.fmt.pix.sizeimage);
       exit(EXIT_FAILURE);
     }

     print_format(&vid_format1);


//open video_device2
     printf("using output device: %s\n", video_device2);
     
     fdwr2 = open(video_device2, O_RDWR);
     assert(fdwr2 >= 0);

     ret_code = ioctl(fdwr2, VIDIOC_QUERYCAP, &vid_caps2);
     assert(ret_code != -1);

     memset(&vid_format2, 0, sizeof(vid_format2));

     ret_code = ioctl(fdwr2, VIDIOC_G_FMT, &vid_format2);

     linewidth2=FRAME_WIDTH2;
     framesize2=FRAME_WIDTH2*FRAME_HEIGHT2*3; // 8x8x8 Bit

     vid_format2.type = V4L2_BUF_TYPE_VIDEO_OUTPUT;
     vid_format2.fmt.pix.width = FRAME_WIDTH2;
     vid_format2.fmt.pix.height = FRAME_HEIGHT2;
     vid_format2.fmt.pix.pixelformat = FRAME_FORMAT2;
     vid_format2.fmt.pix.sizeimage = framesize2;
     vid_format2.fmt.pix.field = V4L2_FIELD_NONE;
     vid_format2.fmt.pix.bytesperline = linewidth2;
     vid_format2.fmt.pix.colorspace = V4L2_COLORSPACE_SRGB;

     // set data format
     ret_code = ioctl(fdwr2, VIDIOC_S_FMT, &vid_format2);
     assert(ret_code != -1);

     print_format(&vid_format2);
}


// unused
void closev4l2()
{
//     close(fdwr0);
     close(fdwr1);
     close(fdwr2);

}

void vframe(char ep[],char EP_error[], int r, int actual_length, unsigned char buf[], unsigned char *colormap) 
{
  // error handler
  time_t now1;
  now1 = time(NULL); 
  if (r < 0) {
    if (strcmp (EP_error, libusb_error_name(r))!=0)
    {       
        strcpy(EP_error, libusb_error_name(r));
        fprintf(stderr, "\n: %s >>>>>>>>>>>>>>>>>bulk transfer (in) %s:%i %s\n", ctime(&now1), ep , r, libusb_error_name(r));
        sleep(1);
    }
    return;
  }
  
  // reset buffer if the new chunk begins with magic bytes or the buffer size limit is exceeded
  unsigned char magicbyte[4]={0xEF,0xBE,0x00,0x00};
  
  if  ((strncmp (buf, magicbyte,4)==0 ) || ((buf85pointer + actual_length) >= BUF85SIZE))
    {
        //printf(">>>>>>>>>>>begin of new frame<<<<<<<<<<<<<\n");
        buf85pointer=0;
    }
 
  //printf("actual_length %d !!!!!\n", actual_length);

  memmove(buf85+buf85pointer, buf, actual_length);
  buf85pointer=buf85pointer+actual_length;
  
  if  ((strncmp (buf85, magicbyte,4)!=0 ))
    {
        //reset buff pointer
        buf85pointer=0;
        printf("Reset buffer because of bad Magic Byte!\n");
        return;
    }
      
  // a quick and dirty job for gcc
  uint32_t FrameSize   = buf85[ 8] + (buf85[ 9] << 8) + (buf85[10] << 16) + (buf85[11] << 24);
  uint32_t ThermalSize = buf85[12] + (buf85[13] << 8) + (buf85[14] << 16) + (buf85[15] << 24);
  uint32_t JpgSize     = buf85[16] + (buf85[17] << 8) + (buf85[18] << 16) + (buf85[19] << 24);
  uint32_t StatusSize  = buf85[20] + (buf85[21] << 8) + (buf85[22] << 16) + (buf85[23] << 24);

  //printf("FrameSize= %d (+28=%d), ThermalSize %d, JPG %d, StatusSize %d, Pointer %d\n",FrameSize,FrameSize+28, ThermalSize, JpgSize,StatusSize,buf85pointer); 

  if ( (FrameSize+28) > (buf85pointer) ) 
  {
    // wait for next chunk
    return;
  }
  
  int i,v;
  // get a full frame, first print the status
  t1=t2;
  gettimeofday(&t2, NULL);
  // fps as moving average over last 20 frames
//  fps_t = (19*fps_t+10000000/(((t2.tv_sec * 1000000) + t2.tv_usec) - ((t1.tv_sec * 1000000) + t1.tv_usec)))/20;

  filecount++;
//  printf("#%08i %lld/10 fps:",filecount,fps_t); 
//  for (i = 0; i <  StatusSize; i++) {
//                    v=28+ThermalSize+JpgSize+i;
//                    if(buf85[v]>31) {printf("%c", buf85[v]);}
//            }
//  printf("\n"); 
  
  buf85pointer=0;
  
  unsigned short pix[160*120];   // display-resampled decoded samples
  unsigned short sensor_pix[160*120]; // original decoded sensor samples
  int sensor_width = 0;
  int sensor_height = 0;
  const char *source_byte_order = "unknown";
  int x, y;
  unsigned char *fb_proc,*fb_proc2; 

  fb_proc = malloc(160 * 128); // 8 Bit gray buffer really needs only 160 x 120
  memset(fb_proc, 128, 160*128);       // sizeof(fb_proc) doesn't work, value depends from LUT
  
  fb_proc2 = malloc(160 * 128 * 3 ); // 8x8x8  Bit RGB buffer 

  int min = 0x10000, max = 0;
  float rms = 0;

  memset(pix, 0, sizeof(pix));
  memset(sensor_pix, 0, sizeof(sensor_pix));

  if (ThermalSize == 10332)
  {
    if (!thermal_layout_reported)
    {
      printf("Thermal layout: FLIR ONE Gen3/Lepton2 80x60 packet stream (%u bytes)\n", ThermalSize);
      thermal_layout_reported = 1;
    }
    decode_lepton2_sensor(&buf85[28], sensor_pix);
    upscale_lepton2_sensor(sensor_pix, pix);
    sensor_width = 80;
    sensor_height = 60;
    source_byte_order = "little";
  }
  else if (ThermalSize >= 39360)
  {
    if (!thermal_layout_reported)
    {
      printf("Thermal layout: FLIR ONE G2 160x120 packet stream (%u bytes)\n", ThermalSize);
      thermal_layout_reported = 1;
    }
    decode_g2_thermal(&buf85[28], sensor_pix);
    memcpy(pix, sensor_pix, sizeof(pix));
    sensor_width = 160;
    sensor_height = 120;
    source_byte_order = "little";
  }

  else
  {
    if (!thermal_layout_reported)
    {
      printf("Unsupported thermal payload size: %u bytes\n", ThermalSize);
      thermal_layout_reported = 1;
    }
  }

  {
    int status_is_ffc = StatusSize >= 20 &&
      strncmp(&buf85[28 + ThermalSize + JpgSize + 17], "FFC", 3) == 0;
    const char *ffc_state = status_is_ffc ? "ffc" : (FFC ? "post_ffc_discarded" : "normal");

    if (sensor_width > 0)
      write_raw_frame(sensor_pix, sensor_width, sensor_height, ThermalSize,
                      source_byte_order, ffc_state);
  }

// Find the max, min and RMS (not used yet) values of the decoded array
  int maxx = 0, maxy = 0;
  for (y = 0; y < 120; ++y) 
  {
    for (x = 0; x < 160; ++x) {
      v = pix[y * 160 + x];
      
      if (v < min) min = v;
      if (v > max) { max = v; maxx = x; maxy = y; }
      rms += v * v;      
    }
  }
    
  // RMS used later
//  rms /= 160 * 120;
//  rms = sqrtf(rms);
  
  for (y = 0; y < 120; ++y)    //120
  {
    for (x = 0; x < 160; ++x) {   //160
      int mapped;

      if (fixed_raw_range_enabled) {
        int range = fixed_raw_high - fixed_raw_low;
        mapped = ((int)pix[y * 160 + x] - fixed_raw_low) * 255 / range;
        if (mapped < 0) mapped = 0;
        if (mapped > 255) mapped = 255;
      } else {
        // Preserve the bridge's existing per-frame min/max display mapping.
        int delta = max - min;
        int scale;
        if (!delta) delta = 1;
        scale = 0x10000 / delta;
        mapped = (pix[y * 160 + x] - min) * scale >> 8;
      }

// fb_proc is the gray scale frame buffer
      fb_proc[y * 160 + x] = mapped;   // unsigned char!!

    }
  }
  
  char st1[100];
  char st2[100];
  struct tm *loctime;
  // Convert it to local time and Print it out in a nice format.
  loctime = localtime (&now1);
  strftime (st1, 60, "%H:%M:%S", loctime);
   
  // calc medium of 2x2 center pixels
  int med = (pix[59 * 160 + 79]+pix[59 * 160 + 80]+pix[60 * 160 + 79]+pix[60 * 160 + 80])/4;
  if (legacy_generic_temperature_overlay)
    sprintf(st2," %.1f/%.1f/%.1f'C", raw2temperature(min),raw2temperature(med),raw2temperature(max));
  else
    sprintf(st2," %d/%d/%d raw", min, med, max);
  strcat(st1, st2);
  
  #define MAX 26 // max chars in line  160/6=26,6 
  strncpy(st2, st1, MAX);
  // write zero to string !! 
  st2[MAX-1] = '\0';
  font_write(fb_proc, 1, 120, st2);

  // show crosshairs, remove if required 
  font_write(fb_proc, 80-2, 60-3, "+");

  maxx -= 4;
  maxy -= 4;

  if (maxx < 0) maxx = 0; 
  if (maxy < 0) maxy = 0;
  if (maxx > 150) maxx = 150;
  if (maxy > 110) maxy = 110;

  font_write(fb_proc, 160-6, maxy, "<");
  font_write(fb_proc, maxx, 120-8, "|");

  for (y = 0; y < 128; ++y) 
  {
    for (x = 0; x < 160; ++x) {  

// fb_proc is the gray scale frame buffer
    v=fb_proc[y * 160 + x] ;   // unsigned char!!

// fb_proc2 is an 24bit RGB buffer

    fb_proc2[3*y * 160 + x*3] = colormap[3 * v];   // unsigned char!!
    fb_proc2[(3*y * 160 + x*3)+1] = colormap[3 * v + 1];   // unsigned char!!
    fb_proc2[(3*y * 160 + x*3)+2] = colormap[3 * v + 2];   // unsigned char!!
    }
  }


    
  // write video to v4l2loopback(s)
//   write(fdwr0, fb_proc, framesize0);  // gray scale Thermal Image
   write(fdwr1, &buf85[28+ThermalSize], JpgSize);  // jpg Visual Image
 
  if (StatusSize >= 20 && strncmp (&buf85[28+ThermalSize+JpgSize+17],"FFC",3)==0)
  {
    FFC=1;  // drop all FFC frames
  } else    
  {        
    if (FFC==1)
    {
      FFC=0; // drop first frame after FFC
    }
    else
    {             
  write_thermal_loopback_frame(fb_proc2, framesize2);  // colorized RGB Thermal Image
    }
    }



  // free memory
  free(fb_proc);                    // thermal RAW
  free(fb_proc2);                   // visible jpg
    
}

 static int find_lvr_flirusb(void)
 {
 	devh = libusb_open_device_with_vid_pid(NULL, VENDOR_ID, PRODUCT_ID);
 	return devh ? 0 : -EIO;
 }
 
 void print_bulk_result(char ep[],char EP_error[], int r, int actual_length, unsigned char buf[])
 {
         time_t now1;
         int i;

         now1 = time(NULL);
         if (r < 0) {
                if (strcmp (EP_error, libusb_error_name(r))!=0)
                {       
                    strcpy(EP_error, libusb_error_name(r));
                    fprintf(stderr, "\n: %s >>>>>>>>>>>>>>>>>bulk transfer (in) %s:%i %s\n", ctime(&now1), ep , r, libusb_error_name(r));
                    sleep(1);
                }
                //return 1;
        } else
        {           
            printf("\n: %s bulk read EP %s, actual length %d\nHEX:\n",ctime(&now1), ep ,actual_length);
            // write frame to file          
  /*
            char filename[100];
            sprintf(filename, "EP%s#%05i.bin",ep,filecount);
            filecount++;
            FILE *file = fopen(filename, "wb");
            fwrite(buf, 1, actual_length, file);
            fclose(file);
  */         
          // hex print of first byte
            for (i = 0; i <  (((200)<(actual_length))?(200):(actual_length)); i++) {
                    printf(" %02x", buf[i]);
            }
                 
            printf("\nSTRING:\n");	
            for (i = 0; i <  (((200)<(actual_length))?(200):(actual_length)); i++) {
                    if(buf[i]>31) {printf("%c", buf[i]);}
            }
            printf("\n");	
            
        } 
 }       
 
 int EPloop(unsigned char *colormap)
 {    
 	int i,r = 1;
 	r = libusb_init(NULL);
 	if (r < 0) {
 		fprintf(stderr, "failed to initialise libusb\n");
 		exit(1);
 	}
 	
  	r = find_lvr_flirusb();
 	if (r < 0) {
 		fprintf(stderr, "Could not find/open device\n");
 		goto out;
 	}
 	printf("Successfully find the Flir One G2 device\n");
	

    r = libusb_set_configuration(devh, 3);
    if (r < 0) {
        fprintf(stderr, "libusb_set_configuration error %d\n", r);
        goto out;
    }
    printf("Successfully set usb configuration 3\n");
	
 
 	// Claiming of interfaces is a purely logical operation; 
    // it does not cause any requests to be sent over the bus. 
 	r = libusb_claim_interface(devh, 0);
 	if (r <0) {
 		fprintf(stderr, "libusb_claim_interface 0 error %d\n", r);
 		goto out;
 	}	
 	r = libusb_claim_interface(devh, 1);
 	if (r < 0) {
 		fprintf(stderr, "libusb_claim_interface 1 error %d\n", r);
 		goto out;
 	}
 	r = libusb_claim_interface(devh, 2);
 	if (r < 0) {
 		fprintf(stderr, "libusb_claim_interface 2 error %d\n", r);
 		goto out;
 	}
 	printf("Successfully claimed interface 0,1,2\n");
	
 	
	unsigned char buf[1048576]; 
    int actual_length;

 	time_t now;
 	// save last error status to avoid clutter the log
	char EP81_error[50]="", EP83_error[50]="",EP85_error[50]=""; 
 	unsigned char data[2]={0,0}; // only a bad dummy
 	
 	// don't forget: $ sudo modprobe v4l2loopback video_nr=0,1
 	startv4l2();
 	
 	int state = 1; 
 	int ct=0;

    while (1)
    {
    	
    switch(state) {
        
         case 1:
            /* Flir config
            01 0b 01 00 01 00 00 00 c4 d5
            0 bmRequestType = 01
            1 bRequest = 0b
            2 wValue 0001 type (H) index (L)    stop=0/start=1 (Alternate Setting)
            4 wIndex 01                         interface 1/2
            5 wLength 00
            6 Data 00 00

            libusb_control_transfer (*dev_handle, bmRequestType, bRequest, wValue,  wIndex, *data, wLength, timeout)
            */
 	
            printf("stop interface 2 FRAME\n");
            r = libusb_control_transfer(devh,1,0x0b,0,2,data,0,100);
            if (r < 0) {
                fprintf(stderr, "Control Out error %d\n", r);
                return r;
            }

            printf("stop interface 1 FILEIO\n");
            r = libusb_control_transfer(devh,1,0x0b,0,1,data,0,100);
            if (r < 0) {
                fprintf(stderr, "Control Out error %d\n", r);
                return r;
            } 
             	
         	printf("\nstart interface 1 FILEIO\n");
         	r = libusb_control_transfer(devh,1,0x0b,1,1,data,0,100);
 	        if (r < 0) {
 		        fprintf(stderr, "Control Out error %d\n", r);
 		        return r;
 	        }
 	        now = time(0); // Get the system time
            printf("\n:xx %s",ctime(&now));
 	        state = 3;   // jump over wait stait 2. Not really using any data from CameraFiles.zip
            break;
        
        
        case 2:
         	printf("\nask for CameraFiles.zip on EP 0x83:\n");     
         	now = time(0); // Get the system time
            printf("\n: %s",ctime(&now));
   
            int transferred = 0;
            char my_string[128];

            //--------- write string: {"type":"openFile","data":{"mode":"r","path":"CameraFiles.zip"}}
            int length = 16;
            unsigned char my_string2[16]={0xcc,0x01,0x00,0x00,0x01,0x00,0x00,0x00,0x41,0x00,0x00,0x00,0xF8,0xB3,0xF7,0x00};
            printf("\nEP 0x02 to be sent Hexcode: %i Bytes[",length);
            int i;
            for (i = 0; i < length; i++) {
                printf(" %02x", my_string2[i]);

            }
            printf(" ]\n");
    
            r = libusb_bulk_transfer(devh, 2, my_string2, length, &transferred, 0);
            if(r == 0 && transferred == length)
            {
                printf("\nWrite successful!");
            }
            else
                printf("\nError in write! res = %d and transferred = %d\n", r, transferred);
    
            strcpy(  my_string,"{\"type\":\"openFile\",\"data\":{\"mode\":\"r\",\"path\":\"CameraFiles.zip\"}}");
    
            length = strlen(my_string)+1;
            printf("\nEP 0x02 to be sent: %s", my_string);
    
            // avoid error: invalid conversion from ‘char*’ to ‘unsigned char*’ [-fpermissive]
            unsigned char *my_string1 = (unsigned char*)my_string;
            //my_string1 = (unsigned char*)my_string;
            
            r = libusb_bulk_transfer(devh, 2, my_string1, length, &transferred, 0);
            if(r == 0 && transferred == length)
            {
                printf("\nWrite successful!");
                printf("\nSent %d bytes with string: %s\n", transferred, my_string);
            }
            else
                printf("\nError in write! res = %d and transferred = %d\n", r, transferred);
 
            //--------- write string: {"type":"readFile","data":{"streamIdentifier":10}}
            length = 16;
            unsigned char my_string3[16]={0xcc,0x01,0x00,0x00,0x01,0x00,0x00,0x00,0x33,0x00,0x00,0x00,0xef,0xdb,0xc1,0xc1};
            printf("\nEP 0x02 to be sent Hexcode: %i Bytes[",length);
            for (i = 0; i < length; i++) {
                printf(" %02x", my_string3[i]);

            }
            printf(" ]\n");
    
            r = libusb_bulk_transfer(devh, 2, my_string3, length, &transferred, 0);
            if(r == 0 && transferred == length)
            {
                printf("\nWrite successful!");
            }
            else
                printf("\nError in write! res = %d and transferred = %d\n", r, transferred);


            //strcpy(  my_string, "{\"type\":\"setOption\",\"data\":{\"option\":\"autoFFC\",\"value\":true}}");
            strcpy(  my_string,"{\"type\":\"readFile\",\"data\":{\"streamIdentifier\":10}}");
            length = strlen(my_string)+1;
            printf("\nEP 0x02 to be sent %i Bytes: %s", length, my_string);
    
            // avoid error: invalid conversion from ‘char*’ to ‘unsigned char*’ [-fpermissive]
            my_string1 = (unsigned char*)my_string;
            
            r = libusb_bulk_transfer(devh, 2, my_string1, length, &transferred, 0);
            if(r == 0 && transferred == length)
            {
                printf("\nWrite successful!");
                printf("\nSent %d bytes with string: %s\n", transferred, my_string);
            }
            else
                printf("\nError in write! res = %d and transferred = %d\n", r, transferred);
 
 
            // go to next state
            now = time(0); // Get the system time
            printf("\n: %s",ctime(&now));
            //sleep(1);
            state = 3;           
            break;
    

        case 3:
         	printf("\nAsk for video stream, start EP 0x85:\n");        

            r = libusb_control_transfer(devh,1,0x0b,1,2,data, 2,200);
            if (r < 0) {
                fprintf(stderr, "Control Out error %d\n", r);
                return r;
            };

            state = 4;
            break;

        case 4:
            // endless loop 
            // poll Frame Endpoints 0x85 
            // don't change timeout=100ms !!
            r = libusb_bulk_transfer(devh, 0x85, buf, sizeof(buf), &actual_length, 100); 
            if (actual_length > 0)
                vframe("0x85",EP85_error, r, actual_length, buf, colormap);
   
            break;      

        }    

        // poll Endpoints 0x81, 0x83
        r = libusb_bulk_transfer(devh, 0x81, buf, sizeof(buf), &actual_length, 10); 
/*
        if (actual_length > 0 && actual_length <= 101)
	{



	char k[5];
	if (strncmp (&buf[32],"VoltageUpdate",13)==0)
	{
	printf("xx %d\n",actual_length);


        char *token, *string, *tofree, *string2;
//	char l;
	strcpy(string,buf);
//       string = buf;
//	 assert(string != NULL);
	printf("yy\n");

        for (i = 32; i <  (((200)<(actual_length))?(200):(actual_length)); i++) 
		{
                    if(string[i]>31) 
		    {
		    printf("%c", string[i]);
//		    printf("%d ", i);
//		    string2[i-32] = string[i];
		    }
		}

           while ((token = strsep(&string, ":")) != NULL)
            {      
	    printf("zz\n");
	    printf("%s\n", token);
	    }

//           free(tofree);
//        for (i = 32; i <  (((200)<(actual_length))?(200):(actual_length)); i++) {
//                    if(buf[i]>31) {printf("%c", buf[i]);}
//            }


	}
	}


*/

        r = libusb_bulk_transfer(devh, 0x83, buf, sizeof(buf), &actual_length, 10); 
        if (strcmp(libusb_error_name(r), "LIBUSB_ERROR_NO_DEVICE")==0) {
 	    	fprintf(stderr, "EP 0x83 LIBUSB_ERROR_NO_DEVICE -> reset USB\n");
    		goto out;
 	    }
//        print_bulk_result("0x83",EP83_error, r, actual_length, buf); 
}
    
    // never reached ;-)
 	libusb_release_interface(devh, 0);
 	
 out:
    //close the device
 	libusb_reset_device(devh);
 	libusb_close(devh);
 	libusb_exit(NULL);
 	return r >= 0 ? r : -r;
 }

static void print_usage(const char *program)
{
  fprintf(stderr,
          "Usage: %s palette.raw [options]\n"
          "  --raw-dir DIRECTORY                 Write decoded sensor samples as .u16le plus JSON metadata\n"
          "  --raw-frame-limit COUNT             Stop raw export after COUNT frames (0 means no limit)\n"
          "  --fixed-raw-low COUNT --fixed-raw-high COUNT\n"
          "                                      Map display colors from a fixed raw-count range\n"
          "  --legacy-generic-temperature-overlay\n"
          "                                      Show the previous unvalidated generic Celsius overlay\n",
          program);
}

static int parse_unsigned_value(const char *text, unsigned long maximum, unsigned long *value)
{
  char *end = NULL;
  unsigned long parsed;

  errno = 0;
  parsed = strtoul(text, &end, 10);
  if (errno != 0 || end == text || *end != '\0' || parsed > maximum)
    return -1;
  *value = parsed;
  return 0;
}

static int configure_raw_output_directory(const char *directory)
{
  if (mkdir(directory, 0755) != 0 && errno != EEXIST) {
    fprintf(stderr, "could not create raw output directory %s: %s\n", directory, strerror(errno));
    return -1;
  }
  raw_output_dir = directory;
  return 0;
}

int main(int argc, char **argv)
{
    int argument_index;
    unsigned long parsed_value;
    unsigned char colormap[768];
    FILE *fp;

	if(argc < 2) {
		print_usage(argv[0]);
		exit(1);
	}

    for (argument_index = 2; argument_index < argc; ++argument_index) {
      if (strcmp(argv[argument_index], "--raw-dir") == 0) {
        if (++argument_index == argc || configure_raw_output_directory(argv[argument_index]) != 0)
          return 1;
      } else if (strcmp(argv[argument_index], "--raw-frame-limit") == 0) {
        if (++argument_index == argc ||
            parse_unsigned_value(argv[argument_index], UINT_MAX, &parsed_value) != 0) {
          fprintf(stderr, "invalid --raw-frame-limit\n");
          return 1;
        }
        raw_frame_limit = (unsigned int)parsed_value;
      } else if (strcmp(argv[argument_index], "--fixed-raw-low") == 0) {
        if (++argument_index == argc ||
            parse_unsigned_value(argv[argument_index], USHRT_MAX, &parsed_value) != 0) {
          fprintf(stderr, "invalid --fixed-raw-low\n");
          return 1;
        }
        fixed_raw_low = (unsigned short)parsed_value;
        fixed_raw_range_enabled = 1;
      } else if (strcmp(argv[argument_index], "--fixed-raw-high") == 0) {
        if (++argument_index == argc ||
            parse_unsigned_value(argv[argument_index], USHRT_MAX, &parsed_value) != 0) {
          fprintf(stderr, "invalid --fixed-raw-high\n");
          return 1;
        }
        fixed_raw_high = (unsigned short)parsed_value;
        fixed_raw_range_enabled = 1;
      } else if (strcmp(argv[argument_index], "--legacy-generic-temperature-overlay") == 0) {
        legacy_generic_temperature_overlay = 1;
      } else {
        fprintf(stderr, "unknown option: %s\n", argv[argument_index]);
        print_usage(argv[0]);
        return 1;
      }
    }

    if (fixed_raw_range_enabled && fixed_raw_low >= fixed_raw_high) {
      fprintf(stderr, "--fixed-raw-low must be less than --fixed-raw-high\n");
      return 1;
    }

    fp = fopen(argv[1], "rb");
    if (fp == NULL) {
      fprintf(stderr, "could not open palette %s: %s\n", argv[1], strerror(errno));
      return 1;
    }
    fread(colormap, sizeof(unsigned char), 768, fp);  // read 256 rgb values
    fclose(fp);

    if (raw_output_dir != NULL)
      printf("raw capture enabled: %s (limit=%u)\n", raw_output_dir, raw_frame_limit);
    if (fixed_raw_range_enabled)
      printf("fixed display mapping enabled: [%u, %u] raw counts\n", fixed_raw_low, fixed_raw_high);

  while (1)
  {
    EPloop(colormap);
  }

  
} 
