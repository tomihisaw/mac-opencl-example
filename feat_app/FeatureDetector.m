//
//  FeatureDetector.m
//  feat_app
//
//  Created by Tom Welsh on 4/26/18.
//  Copyright © 2018 Welsh. All rights reserved.
//

#import "FeatureDetector.h"
#import <Cocoa/Cocoa.h>
#import "OpenCLHelper.h"

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <OpenCL/opencl.h>

#include "kernels_fast9.cl.h"

// Check if a bit is set
bool isSet(uint16_t n, unsigned int d) {
    return n & (1 << (15-d));
}

// For debugging, print binary representation
char* print_binary_16(char *result, uint16_t n) {
    for (int i = 0; i < 16; ++i) {
        result[16-i-1] = ((n >> i) & 1)==1 ? '1' : '0';
    }
    result[16] ='\0';
    return result;
}

// count number of consecutive 1s in circular array
// O(k), k is 16
unsigned int numConsecutive(uint16_t n) {
    unsigned int len = 16;
    unsigned int conseq = 0;
    unsigned int max_conseq = 0;
    
    // scan the back of the array if 1st and last digits are 1
    if (isSet(n, 0) && isSet(n, len-1)) {
        conseq = 2;
        unsigned int i = len-2;
        while (i > 0 && isSet(n, i--))
            conseq++;
    } else {
        conseq = isSet(n, 0);
    }
    unsigned int limited = 16-conseq;
    max_conseq = conseq;
    for (unsigned int i = 1; i < limited; ++i) {
        if (isSet(n, i)) {
            conseq++;
        } else {
            conseq = 0;
        }
        max_conseq = max(conseq, max_conseq);
    }
    return max_conseq;
}

// create lookup table
void create_lookup(uint16_t *results, int size, int k) {
    printf("Creating lookup table ... ");
    for (unsigned i = 0; i < size; ++i) {
        char result[17] = {};
        print_binary_16(result,i);
        int number = numConsecutive((uint16_t)i);
        printf("%d %s: %d\n", i, result, number);
        results[i] = (number > k) ? 1 : 0;
    }
    printf("done\n");
    
}

void testNumber(uint16_t i, unsigned int v) {
    char result[17] = {};
    print_binary_16(result, i);
    int number = numConsecutive(i);
    printf("%d %s: %d\n", i, result, number);
    assert(v==number);
}

void testLookup() {
    testNumber(65535, 16);//1111111111111111
    testNumber(65533, 15);//1111111111111101
    testNumber(65518, 11);//1111111111101110
    testNumber(65452, 9);//1111111110101100
}

@implementation FeatureDetector


// Create constant memory for lookup table
const int SIZE_LU = 65536;
uint16_t lu_in[SIZE_LU];

// Using a singleton so we only load the lookup table once
static FeatureDetector *sharedInstance = nil;
+ (instancetype)sharedInstance {
    static dispatch_once_t once;
    static id sharedInstance;
    dispatch_once(&once, ^{
        sharedInstance = [[self alloc] initPrivate];
    });
    return sharedInstance;
}

- (instancetype)initPrivate {
    
    if (self = [super init]) {
        const char *fname = "lookup.bin";
        // read lookup file from disk if it exists
        // ow create it. To reset, simply delete file
        if( access( fname, F_OK ) != -1 ) {
            printf("Using existing Lookup Table for FAST ...\n");
            FILE *file = fopen(fname, "rb");
            // warning makes an assumption on the file length
            fread(lu_in, SIZE_LU, sizeof(uint16_t), file); // Read in the entire file
            fclose(file); // Close the file
            
            // print location of data (saved two levels up)
            NSArray *paths = NSSearchPathForDirectoriesInDomains(NSApplicationSupportDirectory, NSUserDomainMask, YES);
            NSString *applicationSupportDirectory = [paths firstObject];
            NSLog(@"applicationSupportDirectory: '%@'", applicationSupportDirectory);
        } else { // file doesn't exist, create lookup table
            
            // Create lookup table for bit logic in FAST-9
            printf("Creating Lookup Table for FAST ...\n");
            create_lookup(lu_in, SIZE_LU, 9); // FAST-9
            
            FILE *file = fopen(fname, "wb");
            fwrite(lu_in, sizeof(uint16_t), SIZE_LU, file );
            fclose(file);
        }
    }
    return self;
    
}

// Run the Fast-9 keypoint detector using OpenCL
// Performs non-maximal supression using Harris corner measure
// Draws crosshairs for visualization. Threshold is set by caller, but something like "30" would be reasonable
- (NSImage *)processedImage:(NSImage *)inputImage threshold:(NSUInteger)threshold_in{
    int MAX_POINTS = 65536;
    float threshold = (float) MAX(20,MIN(70,threshold_in));
    int crosshair_width = 5 * (inputImage.size.width/500);
    const int BLOCKSIZE = 128;
    
    // create bitmap data from image in bundle
    NSBitmapImageRep *inputBitmapImageRep = bitmapFromImage(inputImage);
    size_t width_in = inputBitmapImageRep.pixelsWide, height_in = inputBitmapImageRep.pixelsHigh;

    size_t width = (width_in % BLOCKSIZE) == 0 ? width_in : (1+(width_in/BLOCKSIZE))*BLOCKSIZE;
    size_t height = (height_in % BLOCKSIZE) == 0 ? height_in : (1+(height_in/BLOCKSIZE))*BLOCKSIZE;
    printf("width = %zu, height = %zu => %zu x %zu\n", width_in, height_in, width, height);

    unsigned int channels = 4;
    unsigned int * input_pixels = (unsigned int *)inputBitmapImageRep.bitmapData;
    unsigned int * output_pixels = (unsigned int*)malloc(width_in * height_in * sizeof(unsigned int));
    
    cl_image_format format;
    format.image_channel_order = CL_RGBA;
    format.image_channel_data_type = CL_UNSIGNED_INT8;

    cl_mem input_image = gcl_create_image(&format, width_in, height_in, 1, NULL);
    cl_mem output_image = gcl_create_image(&format, width_in, height_in, 1, NULL);
    cl_mem keypoints_buf = gcl_malloc(1 + 2*MAX_POINTS, NULL, CL_MEM_WRITE_ONLY);
    cl_mem keypoints_sup_buf = gcl_malloc(1 + 2*MAX_POINTS, NULL, CL_MEM_WRITE_ONLY);
    cl_mem cornerMeasure_buf = gcl_malloc(MAX_POINTS*sizeof(float), NULL, CL_MEM_WRITE_ONLY);
    cl_mem keypoint_indices = gcl_malloc(width * height * sizeof(unsigned short), NULL, CL_MEM_WRITE_ONLY);

    // CL memory for lookup table
    cl_ushort* table_in  = (cl_ushort*) gcl_malloc(sizeof(cl_ushort) * SIZE_LU, lu_in,
                                                   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR);

    // Get an OpenCL dispatch queue
    dispatch_queue_t queue = create_opencl_dispatch_queue();
    
    clock_t start = clock();
    
    // Dispatch kernel block
    dispatch_sync(queue, ^{

        // Get workgroup size for system
        size_t wgs;
        gcl_get_kernel_block_workgroup_info((__bridge void * _Nonnull)(processImage_kernel),
                                            CL_KERNEL_WORK_GROUP_SIZE,
                                            sizeof(wgs), &wgs, NULL);
        printf("OpenCL determinded workgroup size is %zd.\n", wgs);
        
        // workgroups are 1-pixel high, horizontal slices
        cl_ndrange range = {
            2,                  // The number of dimensions
            {0, 0, 0},          // The offset in each dimension
            {width, height, 0}, // The global range
            {BLOCKSIZE, 1, 0} // The local size of each workgroup
        };
        
        // Copy the image from CPU to GPU
        const size_t origin[3] = { 0, 0, 0 };
        const size_t region[3] = { width_in, height_in, 1};
        
        gcl_copy_ptr_to_image(input_image, input_pixels, origin, region);
        
        fastImageColor_kernel(&range,
                              table_in,
                              input_image,
                              output_image,
                              (cl_int)width_in,
                              (cl_int)height_in,
                              
                              (cl_float)threshold,
                              (cl_int*)keypoints_buf,
                              MAX_POINTS
                              );
        
        harrisCornerMeasure_kernel(&range,
                                   input_image,
                                   (cl_int)width,
                                   (cl_int)height,
                                   (cl_int*)keypoints_buf,
                                   (cl_float*)cornerMeasure_buf,
                                   (cl_ushort*)keypoint_indices
                                   );

        nonmax_Threshold_kernel(&range,
                                (cl_int)width,
                                (cl_int)height,
                                (cl_int*)keypoints_buf,
                                (cl_float*)cornerMeasure_buf,
                                (cl_int*)keypoints_sup_buf,
                                (cl_ushort*)keypoint_indices
                                );


        showKeypoints_kernel(&range,
                             output_image,
                             (cl_int)width_in,
                             (cl_int)height_in,
                             (cl_int)width,
                             (cl_int)crosshair_width,
                             (cl_int*)keypoints_sup_buf
                             );
        

        // Copy back results into pointer
        gcl_copy_image_to_ptr(output_pixels, output_image, origin, region);
        
        gcl_release_image(input_image);
        gcl_release_image(output_image);

        gcl_free(keypoints_buf);
        gcl_free(keypoints_sup_buf);
        gcl_free(cornerMeasure_buf);
        gcl_free(keypoint_indices);

    });
    float worktime = (float) (clock() - start) / CLOCKS_PER_SEC;
    printf("wall time: %f ms\n",worktime*1000);

    //Finally, export to disk
    NSBitmapImageRep * imageRep = [[NSBitmapImageRep alloc] initWithBitmapDataPlanes:(unsigned char **)&output_pixels
                                                                          pixelsWide:width_in
                                                                          pixelsHigh:height_in
                                                                       bitsPerSample:8
                                                                     samplesPerPixel:channels
                                                                            hasAlpha:YES
                                                                            isPlanar:NO
                                                                      colorSpaceName:NSDeviceRGBColorSpace
                                                                        bitmapFormat:NSAlphaNonpremultipliedBitmapFormat
                                                                         bytesPerRow:channels*width_in
                                                                        bitsPerPixel:8*channels];
    
    NSImage *returnImage = [[NSImage alloc] initWithData:[imageRep TIFFRepresentation]];
    free(output_pixels);
    return returnImage;
}
@end
