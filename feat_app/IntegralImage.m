//
//  IntegralImage.m
//  feat_app
//
//  Created by Tom Welsh on 4/30/18.
//  Copyright © 2018 Welsh. All rights reserved.
//

#import "IntegralImage.h"
#import <Cocoa/Cocoa.h>
#import "OpenCLHelper.h"

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <OpenCL/opencl.h>
#include "kernels_integral.cl.h"

@implementation IntegralImage
static IntegralImage *sharedInstance = nil;

// Using a Singleton so initialization can occur once
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
    }
    return self;
}

- (NSImage *)processedImage:(NSImage *)inputImage width:(NSUInteger)blur_radius {

    // Hardcoded block size for the workgroup
    // The image will be upsized to be a multiple of this value
    unsigned int BLOCKSIZE = 128;
    
    // Create bitmap data from image in bundle
    NSBitmapImageRep *inputBitmapImageRep = bitmapFromImage(inputImage);
    size_t width_in = inputBitmapImageRep.pixelsWide,
    height_in = inputBitmapImageRep.pixelsHigh;
    
    // The image is padded (reflection) to accomodate the blur radius and default block size
    size_t width_padded = width_in + 2 * blur_radius;
    size_t height_padded = height_in + 2 * blur_radius;

    size_t width = (width_padded % BLOCKSIZE) == 0 ? width_padded : (1+(width_padded/BLOCKSIZE))*BLOCKSIZE;
    size_t height = (height_padded % BLOCKSIZE) == 0 ? height_padded : (1+(height_padded/BLOCKSIZE))*BLOCKSIZE;
    
    printf("Input: width = %zu, height = %zu => %zu x %zu\n",
           width_in, height_in, width, height);
    
    unsigned int channels = 4;
    unsigned int * input_pixels = (unsigned int *)inputBitmapImageRep.bitmapData;
    unsigned int * output_pixels = (unsigned int*)malloc(width_in * height_in * sizeof(unsigned int));

    // Assumption that the image is coming in as RGBA
    cl_image_format format;
    format.image_channel_order = CL_RGBA;
    format.image_channel_data_type = CL_UNSIGNED_INT8;
    
    cl_mem input_image = gcl_create_image(&format, width_in, height_in, 1, NULL);
    cl_mem output_image = gcl_create_image(&format, width_in, height_in, 1, NULL);
    
    cl_mem temp_buffer = gcl_malloc(width*height*sizeof(float), NULL, CL_MEM_READ_WRITE);
    cl_mem temp_buffer2 = gcl_malloc(width*height*sizeof(float), NULL, CL_MEM_READ_WRITE);
    cl_mem block_sums = gcl_malloc(height*(width/BLOCKSIZE)*sizeof(float), NULL, CL_MEM_READ_WRITE);
    
    // Get an OpenCL dispatch queue
    dispatch_queue_t queue = create_opencl_dispatch_queue();
    
    // Start timing
    clock_t start = clock();
    
    // Dispatch kernel block
    dispatch_sync(queue, ^{
        
        // Get workgroup size for system
        size_t wgs;
        gcl_get_kernel_block_workgroup_info((__bridge void * _Nonnull)(read_luminance_kernel),
                                            CL_KERNEL_WORK_GROUP_SIZE,
                                            sizeof(wgs), &wgs, NULL);
        printf("OpenCL determinded workgroup size is %zd.\n", wgs);
        // Copy the image from CPU to GPU
        const size_t origin[3] = { 0, 0, 0 };
        const size_t region[3] = { width_in, height_in, 1};
        
        gcl_copy_ptr_to_image(input_image, input_pixels, origin, region);
        
        int pad = (int)((width-width_in) / 2);
        cl_ndrange range = {
            2,                  // The number of dimensions
            {0, 0, 0},          // The offset in each dimension
            {width, height, 0}, // The global range
            {BLOCKSIZE, 1, 0}   // The local size of each workgroup
        };
        cl_ndrange range_tr = {
            2,                  // The number of dimensions
            {0, 0, 0},          // The offset in each dimension
            {width, height, 0}, // The global range
            {16, 16, 0}         // The local size of each workgroup
        };
        cl_ndrange range_inv = {
            2,                  // The number of dimensions
            {0, 0, 0},          // The offset in each dimension
            {height, width, 0}, // The global range
            {BLOCKSIZE, 1, 0}   // The local size of each workgroup
        };
        cl_ndrange range_tr_inv = {
            2,                  // The number of dimensions
            {0, 0, 0},          // The offset in each dimension
            {height, width, 0}, // The global range
            {16, 16, 0}         // The local size of each workgroup
        };
        
        read_luminance_kernel(&range,
                         input_image,
                         (cl_float*)temp_buffer,
                         (cl_int)width_in,
                         (cl_int)height_in,
                         (cl_int)width,
                         (cl_int)pad);
        
        scan_kernel(&range,
                    (cl_float*)temp_buffer,
                    (cl_float*)temp_buffer2,
                    (cl_float*)block_sums,
                    (cl_int)width,
                    (cl_int)height);

        add_blocks_kernel(&range,
                          (cl_float*)temp_buffer2,
                          (cl_float*)block_sums,
                          (cl_int)width,
                          (cl_int)height);
        
        transpose_kernel(&range_tr,
                         (cl_float*)temp_buffer2,
                         (cl_float*)temp_buffer,
                         (cl_int)width,
                         (cl_int)height);
        
        
        scan_kernel(&range_inv,
                    (cl_float*)temp_buffer,
                    (cl_float*)temp_buffer2,
                    (cl_float*)block_sums,
                    (cl_int)height,
                    (cl_int)width);

        add_blocks_kernel(&range_inv,
                          (cl_float*)temp_buffer2,
                          (cl_float*)block_sums,
                          (cl_int)height,
                          (cl_int)width);
        
        transpose_kernel(&range_tr_inv,
                         (cl_float*)temp_buffer2,
                         (cl_float*)temp_buffer,
                         (cl_int)height,
                         (cl_int)width);
        
        box_blur_kernel(&range,
                        input_image,
                        output_image,
                        (cl_float*)temp_buffer,
                        (cl_int)blur_radius,
                        (cl_int)width_in,
                        (cl_int)height_in,
                        (cl_int)width,
                        (cl_int)pad);
        
        // Copy back results into pointer
        gcl_copy_image_to_ptr(output_pixels, output_image, origin, region);

        gcl_free(block_sums);
        gcl_free(temp_buffer);
        gcl_free(temp_buffer2);
        gcl_release_image(input_image);
        gcl_release_image(output_image);
    });
    
    float worktime = (float) (clock() - start) / CLOCKS_PER_SEC;
    printf("wall time: %f ms\n", worktime*1000);
    
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
