from PIL import Image

def resize_square(img: Image) -> Image:
    # Determine the shortest side
    min_side = min(img.width, img.height)
    
    # Calculate the left and right crop positions for centering
    left = (img.width - min_side) // 2
    right = left + min_side
    
    # Calculate the top and bottom crop positions for centering
    top = (img.height - min_side) // 2
    bottom = top + min_side
    
    # Crop the image
    img_cropped = img.crop((left, top, right, bottom))
    return img_cropped