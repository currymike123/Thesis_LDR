import numpy as np
from numpy import interp

#HDR FULL PIPELINE

# Multiply Function

def multiply(img):

    # Save the width, height, and channel depth of the image. 
    h = img.shape[0]
    w = img.shape[1]
    d = img.shape[2]

    # Convert image to a float between 0 and 1.
    img = img.astype("float64") / 255

    # Apply the Filter

    # For each pixel row in the image. 
    for y in range(0,h):
        # For each pixel in the row. x = pixel
        for x in range(0,w):
            # For each RGB value.
            for z in range(0,d):
                # Apply Multiply blend
                img[y,x,z] = img[y,x,z] * img[y,x,z]

    # Convert back to an integer between 0 and 255.
    img = img * 255
    img = np.uint8(img)
    
    # Return Multiply blended image
    return img


# Screen Function

def screen(img):

    # Save the width, height, and channel depth of the image. 
    h = img.shape[0]
    w = img.shape[1]
    d = img.shape[2]

    # Convert image to a float between 0 and 1.
    img = img.astype("float64") / 255

    # Apply the Filter

    # For each pixel row in the image. 
    for y in range(0,h):
        # For each pixel in the row. x = pixel
        for x in range(0,w):
            # For each RGB value.
            for z in range(0,d):
                # Apply Screen blend 
                img[y,x,z] = 1 - (1 - img[y,x,z]) * (1 - img[y,x,z])

     # Convert back to an integer between 0 and 255.
    img = img * 255
    img = np.uint8(img)

    # Return Screen blended image
    return img

### HDR function

def hdr(img):

    # Save the width, height, and channel depth of the image. 
    h = img.shape[0]
    w = img.shape[1]
    d = img.shape[2]

    # Convert images to a float between 0 and 1.  Create blended images and a copy of the original image. 

    blendMult = multiply(img).astype("float64") / 255
    blendScreen = screen(img).astype("float64") / 255
    img = img.astype("float64") / 255
    imgBlend = img

    # For each pixel row in the image. 

    for y in range(0,h):
        # For each pixel in the row. x = pixel
        for x in range(0,w):
            # For each RGB value.
            for z in range(0,d):

                # Preceived brightness 
                brightness = (img[y,x,0] * 0.2126) + (img[y,x,1] * 0.7152) + (img[y,x,2] * 0.0722)
                
                # Set a high and low cut off for the multiply (cuttOffHigh) and screen (cuttOffLow) images. 
                cutOffHigh = 2/3
                cutOffLow = 1/3
                
                if(brightness > cutOffHigh):

                    # Calculate an alpha mask so the blending fades out the closer you get to the midtone values.
                    alpha = np.interp(brightness,[cutOffHigh,1],[0,1])

                    # Blend the mulitplied image
                    imgBlend[y,x,z] = (img[y,x,z] * (1-alpha)) + (blendMult[y,x,z] * (alpha))

                if(brightness < cutOffLow):

                    # Calculate an alpha mask so the blending fades out the closer you get to the midtone values.
                    alpha = np.interp(brightness,[0,cutOffLow],[1,0])

                    #  Blend the screened image
                    imgBlend[y,x,z] = (img[y,x,z] * (1-alpha)) + (blendScreen[y,x,z] * (alpha))
                
    # Convert back to an integer between 0 and 255.
    imgBlend = imgBlend * 255
    imgBlend = np.uint8(imgBlend)


    # Return the blended image. 
    return imgBlend