//
//  IntegralImage.h
//  feat_app
//
//  Created by Tom Welsh on 4/30/18.
//  Copyright © 2018 Welsh. All rights reserved.
//

#import <Foundation/Foundation.h>

@interface IntegralImage : NSObject
+ (instancetype)sharedInstance;
- (NSImage *)processedImage:(NSImage *)inputImage width:(NSUInteger)blur_radius;

@end
