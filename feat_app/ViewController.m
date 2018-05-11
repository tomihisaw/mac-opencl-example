//
//  ViewController.m
//  feat_app
//
//  Created by Tom Welsh on 4/26/18.
//  Copyright © 2018 Welsh. All rights reserved.
//

#import "ViewController.h"
#import "FeatureDetector.h"
#import "IntegralImage.h"

@implementation ViewController {
    NSImage *inputImage;
    NSImage *ex1;
    NSImage *ex2;
}

- (IBAction)checkerboardSelected:(id)sender {
    printf("checker");
    inputImage = ex1;
    [self updateImage];
}

- (IBAction)flowerSelected:(id)sender {
    printf("flower");
    inputImage = ex2;
    [self updateImage];
}

- (IBAction)blurSliderChanged:(id)sender {
    
    int sliderValue = [sender intValue];
    printf("Slider Value %d\n",sliderValue);
    
    [self updateImage];
}

- (IBAction)blurSelected:(id)sender {
    
    [self updateImage];
}

- (void) updateImage {
    
    if (1 == self.blurRadio.integerValue) {
        self.imView.image =  [[IntegralImage sharedInstance] processedImage:inputImage width:self.blurSlider.integerValue];
    } else if (1 == self.featRadio.integerValue) {
        self.imView.image = [[FeatureDetector sharedInstance] processedImage:inputImage threshold:self.blurSlider.integerValue];
    }
}

- (void)viewDidLoad {
    [super viewDidLoad];
    
    self.blurSlider.integerValue = 20;
    ex2 = [NSImage imageNamed:@"flower.jpg"];
    ex1 = [NSImage imageNamed:@"checkerboard.jpg"];
    inputImage = ex1;
    
    [self updateImage];
    
}

- (void)setRepresentedObject:(id)representedObject {
    [super setRepresentedObject:representedObject];
}


@end
