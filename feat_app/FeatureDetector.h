//
//  FeatureDetector.h
//  feat_app
//
//  Created by Tom Welsh on 4/26/18.
//  Copyright © 2018 Welsh. All rights reserved.
//

#import <Foundation/Foundation.h>

@interface FeatureDetector : NSObject
+ (instancetype)sharedInstance;
- (NSImage *)processedImage:(NSImage *)inputImage threshold:(NSUInteger)threshold;
@end
