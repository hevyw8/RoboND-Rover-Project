import numpy as np
import cv2

# Identify pixels above the threshold
# Threshold of RGB > 160 does a nice job of identifying ground pixels only
def color_thresh(img, rgb_thresh=(160, 160, 160)):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    above_thresh = (img[:,:,0] > rgb_thresh[0]) \
                & (img[:,:,1] > rgb_thresh[1]) \
                & (img[:,:,2] > rgb_thresh[2])
    # Index the array of zeros with the boolean array and set to 1
    color_select[above_thresh] = 1
    # Return the binary image
    return color_select

def obs_thresh(img, rgb_thresh=(160,160,160)):
    obs_select = np.zeros_like(img[:,:,0])
    
    below_thresh = (img[:,:,0] < rgb_thresh[0]) \
                & (img[:,:,1] < rgb_thresh[1]) \
                & (img[:,:,2] < rgb_thresh[2])
            
    obs_select[below_thresh] = 1
    return obs_select

def rock_thresh(img, high_thresh=(210,190,50), low_thresh=(110,100,0)):
    rock_select = np.zeros_like(img[:,:,0])
    
    in_thresh = (img[:,:,0] < high_thresh[0]) \
              & (img[:,:,0] > low_thresh[0]) \
              & (img[:,:,1] < high_thresh[1]) \
              & (img[:,:,1] > low_thresh[1]) \
              & (img[:,:,2] < high_thresh[2]) \
              & (img[:,:,2] > low_thresh[2]) 
    
    rock_select[in_thresh] = 1
    return rock_select

# Define a function to convert to rover-centric coordinates
def rover_coords(binary_img):
    # Identify nonzero pixels
    ypos, xpos = binary_img.nonzero()
    # Calculate pixel positions with reference to the rover position being at the 
    # center bottom of the image.  
    x_pixel = np.absolute(ypos - binary_img.shape[0]).astype(np.float)
    y_pixel = -(xpos - binary_img.shape[0]).astype(np.float)
    return x_pixel, y_pixel


# Define a function to convert to radial coords in rover space
def to_polar_coords(x_pixel, y_pixel):
    # Convert (x_pixel, y_pixel) to (distance, angle) 
    # in polar coordinates in rover space
    # Calculate distance to each pixel
    dist = np.sqrt(x_pixel**2 + y_pixel**2)
    # Calculate angle away from vertical for each pixel
    angles = np.arctan2(y_pixel, x_pixel)
    return dist, angles

# Define a function to apply a rotation to pixel positions
def rotate_pix(xpix, ypix, yaw):
    # Convert yaw to radians
    # Apply a rotation
    yaw_rad = yaw * np.pi / 180
    xpix_rotated = xpix * np.cos(yaw_rad) - ypix * np.sin(yaw_rad)
    ypix_rotated = xpix * np.sin(yaw_rad) + ypix * np.cos(yaw_rad)
    # Return the result  
    return xpix_rotated, ypix_rotated

# Define a function to perform a translation
def translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale):
    # Apply a scaling and a translation
    # Assume a scale factor of 10 between world space pixels and rover space pixels
    scale = 10
    # Perform translation and convert to integer since pixel values can't be float
    xpix_translated = np.int_(xpos + (xpix_rot / scale))
    ypix_translated = np.int_(ypos + (ypix_rot / scale))
    # Return the result  
    return xpix_translated, ypix_translated

# Define a function to apply rotation and translation (and clipping)
# Once you define the two functions above this function should work
def pix_to_world(xpix, ypix, xpos, ypos, yaw, world_size, scale):
    # Apply rotation
    xpix_rot, ypix_rot = rotate_pix(xpix, ypix, yaw)
    # Apply translation
    xpix_tran, ypix_tran = translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale)
    # Perform rotation, translation and clipping all at once
    x_pix_world = np.clip(np.int_(xpix_tran), 0, world_size - 1)
    y_pix_world = np.clip(np.int_(ypix_tran), 0, world_size - 1)
    # Return the result
    return x_pix_world, y_pix_world

# Define a function to perform a perspective transform
def perspect_transform(img, src, dst):
           
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))# keep same size as input image
    
    return warped


# Apply the above functions in succession and update the Rover state accordingly
def perception_step(Rover):
    # Perform perception steps to update Rover()
    # TODO: 
    # NOTE: camera image is coming to you in Rover.img
    # 1) Define source and destination points for perspective transform
    dst_size = 5 
    bottom_offset = 4
    source = np.float32([[14, 140], [301 ,140],[200, 96], [118, 96]])
    destination = np.float32([[Rover.img.shape[1]/2 - dst_size, Rover.img.shape[0] - bottom_offset],
                 [Rover.img.shape[1]/2 + dst_size, Rover.img.shape[0] - bottom_offset],
                 [Rover.img.shape[1]/2 + dst_size, Rover.img.shape[0] - 2*dst_size - bottom_offset], 
                 [Rover.img.shape[1]/2 - dst_size, Rover.img.shape[0] - 2*dst_size - bottom_offset]])
    # 2) Apply perspective transform
    warped = perspect_transform(Rover.img, source, destination)
    
    
    # 3) Apply color threshold to identify navigable terrain/obstacles/rock samples
    nav_warped = color_thresh(warped)
    obs_warped = obs_thresh(warped)
    rock_warped = rock_thresh(warped)
    # 4) Update Rover.vision_image (this will be displayed on left side of screen)
    Rover.vision_image[:,:,0] = obs_warped
    Rover.vision_image[:,:,1] = rock_warped
    Rover.vision_image[:,:,2] = nav_warped

    # 5) Convert map image pixel values to rover-centric coords
    nav_xpix, nav_ypix = rover_coords(nav_warped)
    obs_xpix, obs_ypix = rover_coords(obs_warped)
    rock_xpix, rock_ypix = rover_coords(rock_warped)
    # 6) Convert rover-centric pixel values to world coordinates
    scale = 10
    nav_xpix_world, nav_ypix_world = pix_to_world(nav_xpix, nav_ypix, 
                                                  Rover.pos[0],
                                                  Rover.pos[1],
                                                  Rover.yaw,
                                                  Rover.worldmap.shape[0],
                                                  scale)
    obs_xpix_world, obs_ypix_world = pix_to_world(obs_xpix, obs_ypix, 
                                                  Rover.pos[0],
                                                  Rover.pos[1],
                                                  Rover.yaw,
                                                  Rover.worldmap.shape[0],
                                                  scale)
    rock_xpix_world, rock_ypix_world = pix_to_world(rock_xpix, rock_ypix, 
                                                  Rover.pos[0],
                                                  Rover.pos[1],
                                                  Rover.yaw,
                                                  Rover.worldmap.shape[0],
                                                  scale)
    # 7) Update Rover worldmap (to be displayed on right side of screen)
    Rover.worldmap[obs_ypix_world, obs_xpix_world, 0] += 255
    Rover.worldmap[rock_ypix_world, rock_xpix_world, 1] += 255
    Rover.worldmap[nav_ypix_world, nav_xpix_world, 2] += 255

    # 8) Convert rover-centric pixel positions to polar coordinates
    Rover.nav_dists, Rover.nav_angles = to_polar_coords(nav_xpix, nav_ypix) 
    # Update Rover pixel distances and angles
        # Rover.nav_dists = rover_centric_pixel_distances
        # Rover.nav_angles = rover_centric_angles
    
    return Rover