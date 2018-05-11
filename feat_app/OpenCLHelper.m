//
//  OpenCLHelper.m
//
//  Created by Tom Welsh on 4/26/18.
//  Copyright © 2018 Welsh. All rights reserved.
//

#import <Foundation/Foundation.h>
#import <Cocoa/Cocoa.h>
#import <OpenCL/opencl.h>

// Gets bitmap data from an NSImage.
// promotes image file to RGBA from RGB to work better with OpenCL
// TODO: this could be optimized
NSBitmapImageRep *bitmapFromImage(NSImage *srcImage) {
    
    CGImageSourceRef source = CGImageSourceCreateWithData((__bridge CFDataRef)[srcImage TIFFRepresentation], NULL);
    CGImageRef imageRef =  CGImageSourceCreateImageAtIndex(source, 0, NULL);
    CGRect rect = CGRectMake(0.f, 0.f, CGImageGetWidth(imageRef), CGImageGetHeight(imageRef));
    unsigned char *data = calloc(rect.size.width * 4 * rect.size.height, sizeof(unsigned char));
    CGContextRef bitmapContext = CGBitmapContextCreate(data,
                                                       rect.size.width,
                                                       rect.size.height,
                                                       CGImageGetBitsPerComponent(imageRef),
                                                       rect.size.width * 4,
                                                       CGImageGetColorSpace(imageRef),
                                                       kCGImageAlphaPremultipliedLast
                                                       );
    
    CGContextDrawImage(bitmapContext, rect, imageRef);
    
    CGImageRef decompressedImageRef = CGBitmapContextCreateImage(bitmapContext);
    
    NSImage *finalImage = [[NSImage alloc] initWithCGImage:decompressedImageRef size:NSZeroSize];
    NSData *imageData = [finalImage  TIFFRepresentation];
    NSBitmapImageRep *imageRep = [NSBitmapImageRep imageRepWithData:imageData];

    CFRelease(source);
    CFRelease(imageRef);
    free(data);
    CGImageRelease(decompressedImageRef);
    CGContextRelease(bitmapContext);
    
    return imageRep;
}

// Display some info on the device being used
void display_device(cl_device_id device) {
    char name_buf[128];
    char vendor_buf[128];
    
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(char)*128, name_buf, NULL);
    clGetDeviceInfo(device, CL_DEVICE_VENDOR, sizeof(char)*128, vendor_buf, NULL);
    
    fprintf(stdout, "Using OpenCL device: %s %s\n", vendor_buf, name_buf);
}

// Attempt to create a GPU based dispatch queue, falls back to CPU
dispatch_queue_t create_opencl_dispatch_queue(void) {
    
    // Attempt to create GPU based OpenCL dispatch queue
    dispatch_queue_t queue = gcl_create_dispatch_queue(CL_DEVICE_TYPE_GPU,
                                                       NULL);
    if (!queue) {
        fprintf(stdout, "Unable to create a GPU-based dispatch queue.\n");
    }
    
    // Fallback to CPU dispatch queue
    if (queue == NULL) {
        queue = gcl_create_dispatch_queue(CL_DEVICE_TYPE_CPU, NULL);
    }
    
    if (!queue) {
        fprintf(stdout, "Unable to create a CPU-based dispatch queue.\n");
        assert(false);
    }

    cl_device_id device = gcl_get_device_id_with_dispatch_queue(queue);

    // For debugging, we print so we can verify that we are using the GPU
    display_device(device);
    return queue;
}
