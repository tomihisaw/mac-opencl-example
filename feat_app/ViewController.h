//
//  ViewController.h
//  feat_app
//
//  Created by Tom Welsh on 4/26/18.
//  Copyright © 2018 Welsh. All rights reserved.
//

#import <Cocoa/Cocoa.h>

@interface ViewController : NSViewController
@property (weak) IBOutlet NSImageView *imView;
@property (weak) IBOutlet NSButton *blurRadio;
@property (weak) IBOutlet NSButton *featRadio;

@property (weak) IBOutlet NSSliderCell *blurSlider;

@end

