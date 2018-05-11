//
//  OpenCLHelper.h
//
//  Created by Tom Welsh on 4/26/18.
//  Copyright © 2018 Welsh. All rights reserved.
//

#ifndef Helper_h
#define Helper_h

#include <OpenCL/opencl.h>

#ifndef max
#define max(a,b) ((a) > (b) ? (a) : (b))
#endif

NSBitmapImageRep *bitmapFromImage(NSImage *srcImage);
void display_device(cl_device_id device);
dispatch_queue_t create_opencl_dispatch_queue(void);

#endif /* Helper_h */
