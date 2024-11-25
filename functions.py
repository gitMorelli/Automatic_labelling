import cv2
import numpy as np
from google.cloud import documentai_v1 as documentai
from google.cloud import vision_v1 as vision
from google.api_core.client_options import ClientOptions
from google.oauth2 import service_account
import os
from skimage.metrics import structural_similarity as compare_ssim
from skimage.measure import shannon_entropy
from skimage.feature import graycomatrix, graycoprops
import random

def recognize_handwritten_text(client,cropped_image):

    # Encode the cropped image into memory (convert to binary)
    _, encoded_image = cv2.imencode('.jpg', cropped_image)
    content = encoded_image.tobytes()

    image = vision.Image(content=content)

    # Perform text detection
    response = client.document_text_detection(image=image)

    if response.error.message:
        raise Exception(f"Error during API call: {response.error.message}")

    # Extract and print the detected text
    if response.text_annotations:
        detected_text = response.text_annotations[0].description
        print("Detected Text:")
        print(detected_text)
        return detected_text
    else:
        print("No text detected in the cropped image.")
        return None


def superimpose_images(image1_path, image2_path, file_name):
    # Load the two images
    image1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)

    # Ensure both images have the same dimensions
    if image1.shape != image2.shape:
        #print(image1.shape, image2.shape)   
        raise ValueError("The images must have the same dimensions to be superimposed.")

    # Superimpose the images by averaging their pixel values
    superimposed_image = cv2.addWeighted(image1, 0.5, image2, 0.5, 0)

    # Save the superimposed image
    output_path = f"analysis_steps\{file_name}"
    cv2.imwrite(output_path, superimposed_image)
    print(f"Superimposed image saved as {output_path}")

def match_image_dimensions(filled_template, template):
    # Get dimensions of the template
    template_height, template_width = template.shape[0], template.shape[1]

    # Resize the filled template to match the dimensions of the template
    resized_filled_template = cv2.resize(filled_template, (template_width, template_height), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite("analysis_steps\\resized_filled_template.png", resized_filled_template)
    return resized_filled_template

def align_to_template(resized_filled_template, template):
    # Step 1: Detect ORB features and descriptors
    orb = cv2.ORB_create()
    keypoints1, descriptors1 = orb.detectAndCompute(template, None)
    keypoints2, descriptors2 = orb.detectAndCompute(resized_filled_template, None)

    # Step 2: Match features using the BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)

    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Step 3: Select good matches and extract matching points
    good_matches = matches[:50]  # Use the top 50 matches for alignment
    points1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    points2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Step 4: Find homography matrix to align images
    H, mask = cv2.findHomography(points2, points1, cv2.RANSAC, 5.0)

    # Step 5: Warp the filled template to align with the template
    aligned_filled_template = cv2.warpPerspective(resized_filled_template, H, (template.shape[1], template.shape[0]))

    # Optional: Save or display the aligned image
    cv2.imwrite("analysis_steps\\aligned_filled_template.png", aligned_filled_template)
    return aligned_filled_template

def mask_from_template(template):
    # Step 1: Threshold the template to create a binary mask
    # Set all non-white pixels to 1 (black) and white pixels to 0 (background)
    _, mask = cv2.threshold(template.copy(), 240, 1, cv2.THRESH_BINARY_INV)

    # Step 3: Apply dilation on the inverted mask to thicken black regions
    kernel_size = 5  # Adjust the kernel size if needed
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated_mask = cv2.dilate(mask, kernel, iterations=1)
    return dilated_mask

def apply_mask(aligned_filled_template, dilated_mask):
    _, mask_filled = cv2.threshold(aligned_filled_template.copy(), 240, 1, cv2.THRESH_BINARY_INV)
    kernel_size = 5  # Adjust the kernel size if needed
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated_mask_filled = cv2.dilate(mask_filled, kernel, iterations=1)
    # Step 5: Apply the dilated mask to the filled template
    masked_filled_template = aligned_filled_template.copy()
    masked_filled_template[dilated_mask*dilated_mask_filled == 1] = 0  # Set to 0 to cover the area
    masked_filled_template_inverted=255-masked_filled_template*255
    cv2.imwrite("analysis_steps/masked_filled_template_inverted.png", masked_filled_template_inverted)
    cv2.imwrite("analysis_steps/dilated_mask.png", dilated_mask * 255)
    return masked_filled_template_inverted

def darken_roi(masked_filled_template_inverted):
    masked_filled_template=masked_filled_template_inverted.copy()
    kernel_size = 10  # Adjust the kernel size if needed
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    final = cv2.dilate(1-masked_filled_template.copy(), kernel, iterations=3)
    cv2.imwrite("dark_roi.png", final)
    return final

def process_document(file_path,processor_id,project_id,location,client):
    processor_name = f"projects/{project_id}/locations/{location}/processors/{processor_id}"
    
    # Load the image content
    with open(file_path, "rb") as f:
        file_content = f.read()
    
    # Set MIME type based on file type (adjust if necessary)
    document = {"content": file_content, "mime_type": "image/png"}
    request = {"name": processor_name, "raw_document": document}
    
    # Process the document
    result = client.process_document(request=request)
    document = result.document
    return result,document

def show_checkmarks(file_path, document,color=(255,0,0),bounding_boxes=[]):
    # Load the image
    image = cv2.imread(file_path)
    
    # Convert to RGB (OpenCV loads images in BGR format by default)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Iterate over each paragraph
    for page in document.pages:
        for visual_element in page.visual_elements:
            # Get bounding box vertices
            vertices = [(int(vertex.x), int(vertex.y)) for vertex in visual_element.layout.bounding_poly.vertices]
            
            if visual_element.type_ == "filled_checkbox":
                # Draw the bounding box with red color (RGB: 255, 0, 0)
                pts = np.array(vertices, np.int32)
                pts = pts.reshape((-1, 1, 2))  # Required format for cv2.polylines
                cv2.polylines(image, [pts], isClosed=True, color=color, thickness=2)
                bounding_boxes.append([vertices, "filled_checkbox"])
    
    # Convert the image back to BGR for OpenCV to handle displaying properly
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Optionally, save the image with bounding boxes
    output_path = "analysis_steps\output_checkmarks.png"
    cv2.imwrite(output_path, image)
    print(f"Image with checkmarks saved as {output_path}")
    return bounding_boxes

def show_tokens(file_path, document,color=(0,255,0),bounding_boxes=[],file_name="output_text.png"):
    # Load the image using OpenCV
    image = cv2.imread(file_path)
    
    # Iterate over each paragraph
    for page in document.pages:
        for token in page.tokens:
            # Get bounding box vertices
            vertices = [(vertex.x, vertex.y) for vertex in token.layout.bounding_poly.vertices]
            pts = np.array(vertices, dtype=np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(image, [pts], isClosed=True, color=color, thickness=2)
            bounding_boxes.append([vertices, "to_determine"])
    
    # Optionally, save the image with bounding boxes
    output_path = f"analysis_steps\{file_name}"
    cv2.imwrite(output_path, image)
    print(f"Image with handwritten text saved as {output_path}")
    return bounding_boxes

def show_paragraphs(file_path, document, bounding_boxes=[],color=(0,255,0),file_name="paragraphs_on_original.png"):
    # Load the image using OpenCV
    image = cv2.imread(file_path)
    
    # Iterate over each paragraph
    for page in document.pages:
        for paragraph in page.paragraphs:
            # Get bounding box vertices
            vertices = [(vertex.x, vertex.y) for vertex in paragraph.layout.bounding_poly.vertices]
            pts = np.array(vertices, dtype=np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(image, [pts], isClosed=True, color=color, thickness=2)
            bounding_boxes.append([vertices, "to_determine"])
    
    # Optionally, save the image with bounding boxes
    output_path = f"analysis_steps\{file_name}"
    cv2.imwrite(output_path, image)
    return bounding_boxes
def magnify_bb(vertices,magnification=1.0):
    vertices = np.array(vertices, dtype="float32")
    center=np.array([np.mean(vertices[:,0]),np.mean(vertices[:,1])])
    relative_vertices=vertices-center
    relative_vertices*=magnification
    vertices=center+relative_vertices
    vertices = vertices.astype(int)
    return [(v[0],v[1]) for v in vertices]
def preprocess_bounding_box(image, vertices, operation="binarization"):
    vertices = np.array(vertices, dtype="float32")
    width = int(max(
        np.linalg.norm(vertices[0] - vertices[1]),  # Distance between top-left and top-right
        np.linalg.norm(vertices[2] - vertices[3])   # Distance between bottom-left and bottom-right
    ))
    height = int(max(
        np.linalg.norm(vertices[0] - vertices[3]),  # Distance between top-left and bottom-left
        np.linalg.norm(vertices[1] - vertices[2])   # Distance between top-right and bottom-right
    ))

    # Step 2: Define the destination points for the perspective transform
    destination_points = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype="float32")

    # Step 3: Compute the perspective transform matrix
    transform_matrix = cv2.getPerspectiveTransform(np.array(vertices, dtype="float32"), destination_points)

    # Step 4: Warp the image to obtain a top-down view of the region
    warped_region = cv2.warpPerspective(image, transform_matrix, (width, height))

    if operation=="none":
        return warped_region

    # Step 5: Convert to grayscale
    if len(warped_region.shape) == 3:  # Check if the image is not grayscale
        warped_region = cv2.cvtColor(warped_region, cv2.COLOR_BGR2GRAY)

    if operation == "binarization":
        _, binary_region = cv2.threshold(warped_region, 240, 1, cv2.THRESH_BINARY)
        return binary_region
    elif operation == "binarization_cc":
        _, binary_region = cv2.threshold(warped_region, 125, 255, cv2.THRESH_BINARY_INV)
        #binary_region = cv2.bitwise_not(binary_region)
        kernel_size = 5  # Adjust the kernel size if needed
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        dilated_region = cv2.dilate(binary_region, kernel, iterations=3)
        return dilated_region #255*(1-dilated_region)
    elif operation == "binarization_average":
        _, binary_region = cv2.threshold(warped_region, 128, 1, cv2.THRESH_BINARY_INV)
        #binary_region = cv2.bitwise_not(binary_region)
        kernel_size = 5  # Adjust the kernel size if needed
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        dilated_region = cv2.dilate(binary_region, kernel, iterations=1)
        return dilated_region #255*(1-dilated_region)
    elif operation == "orb_preprocessing":
        smoothed_image = cv2.GaussianBlur(warped_region, (5, 5), 0)
        return smoothed_image
    elif operation == "gray":
        return warped_region

def process_bounding_box(image,template, vertices, file_name="None", operation="average",index=0):
    """
    Process the image region defined by the four vertices of a bounding box,
    binarize it, and compute the average pixel value.

    Parameters:
    - image (numpy array): The input image.
    - vertices (numpy array): A 4x2 numpy array or list of points [(x1, y1), (x2, y2), (x3, y3), (x4, y4)] 
                              defining the corners of the bounding box.

    Returns:
    - float: The average value of the binarized pixels in the region.
    """
    # Step 1: Define the target dimensions for the region (rectangular ROI)
    # Ensure vertices is a NumPy array

    if operation == "average":
        template_preprocessed=preprocess_bounding_box(template, vertices,"binarization_average")
        image_preprocessed = preprocess_bounding_box(image, vertices,"binarization_average")
        diff= np.mean(image_preprocessed) / (np.mean(template_preprocessed)+1e-10)
        '''if index==224:
            random_integer = random.randint(1, 100)
            cv2.imwrite(f"Frames\\{random_integer}_frame_template.png",preprocess_bounding_box(template, vertices,"none"))
            cv2.imwrite(f"Frames\\{random_integer}_frame_filled.png",preprocess_bounding_box(image, vertices,"none"))
            cv2.imwrite(f"Frames\\{random_integer}_frame_template_bin.png",preprocess_bounding_box(template, vertices,"binarization_average")*255)
            cv2.imwrite(f"Frames\\{random_integer}_frame_filled_bin.png",preprocess_bounding_box(image, vertices,"binarization_average")*255)
            #print(f"Image with bounding boxes saved as {file_name}")'''
        return diff
    elif operation == "cc":
        template_preprocessed=preprocess_bounding_box(template, vertices,"binarization_cc")
        image_preprocessed = preprocess_bounding_box(image, vertices,"binarization_cc")
        num_labels_template, _ = cv2.connectedComponents(template_preprocessed,connectivity=8)
        num_labels_image, _ = cv2.connectedComponents(image_preprocessed,connectivity=8)
        diff = num_labels_image - num_labels_template
        '''if diff>=2:
            random_integer = random.randint(1, 100)
            cv2.imwrite(f"Frames\\{random_integer}_frame_template.png",preprocess_bounding_box(template, vertices,"none"))
            cv2.imwrite(f"Frames\\{random_integer}_frame_filled.png",preprocess_bounding_box(image, vertices,"none"))
            cv2.imwrite(f"Frames\\{random_integer}_frame_template_bin.png",preprocess_bounding_box(template, vertices,"binarization_cc"))
            cv2.imwrite(f"Frames\\{random_integer}_frame_filled_bin.png",preprocess_bounding_box(image, vertices,"binarization_cc"))
            #print(f"Image with bounding boxes saved as {file_name}")'''
        return diff
    elif operation == "ssim":
        template_preprocessed=preprocess_bounding_box(template, vertices, operation="gray")
        image_preprocessed = preprocess_bounding_box(image, vertices, operation="gray")
        min_size=np.min(image_preprocessed.shape)
        if  min_size<=7:
            score = compare_ssim(template_preprocessed, image_preprocessed,win_size=min_size-1-(min_size%2))
        else:
            score = compare_ssim(template_preprocessed, image_preprocessed)
        return -score #between -1 and 1 (1 is perfect match)
    elif operation == "histogram": #it seems pretty useless
        template_preprocessed=preprocess_bounding_box(template, vertices, operation="none")
        image_preprocessed = preprocess_bounding_box(image, vertices, operation="none")

        # Calculate histograms
        hist1 = cv2.calcHist([template_preprocessed], [0], None, [256], [0, 256])  # Grayscale histogram
        hist2 = cv2.calcHist([image_preprocessed], [0], None, [256], [0, 256])  # Grayscale histogram

        # Normalize histograms
        hist1 = cv2.normalize(hist1, hist1).flatten()
        hist2 = cv2.normalize(hist2, hist2).flatten()

        # Compare histograms using correlation
        score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        return score #score=1 is perfect correlation
    elif operation == "orb":
        template_preprocessed=preprocess_bounding_box(template, vertices, operation="orb_preprocessing")
        image_preprocessed = preprocess_bounding_box(image, vertices, operation="orb_preprocessing")
        # Initialize ORB detector
        '''orb = cv2.ORB_create(nfeatures=100,scaleFactor=1.2)

        # Detect key points and compute descriptors
        keypoints1, descriptors1 = orb.detectAndCompute(template_preprocessed, None)
        keypoints2, descriptors2 = orb.detectAndCompute(image_preprocessed, None)'''

        # Match features using BFMatcher
        '''bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.match(descriptors1, descriptors2)

        # Sort matches by distance
        matches = sorted(matches, key=lambda x: x.distance)
        print(matches)'''
        # Step 1: Detect SIFT features and descriptors
        sift = cv2.SIFT_create()
        keypoints1, descriptors1 = sift.detectAndCompute(template_preprocessed, None)
        keypoints2, descriptors2 = sift.detectAndCompute(image_preprocessed, None)

        '''surf = cv2.xfeatures2d.SURF_create()  # Requires OpenCV contrib package
        keypoints1, descriptors1 = surf.detectAndCompute(template_preprocessed, None)
        keypoints2, descriptors2 = surf.detectAndCompute(image_preprocessed, None)'''

        # Step 2: Match features using the BFMatcher with L2 norm (for SIFT)
        #bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        #matches = bf.match(descriptors1, descriptors2)
        diff=-len(keypoints1)+len(keypoints2)

        return diff

    elif operation == "entropy":
        template_preprocessed=preprocess_bounding_box(template, vertices, operation="gray")
        image_preprocessed = preprocess_bounding_box(image, vertices, operation="gray")
        entropy1 = shannon_entropy(template_preprocessed)
        entropy2 = shannon_entropy(image_preprocessed)
        return -abs(entropy1 / entropy2)
    elif operation == "contours":
        template_preprocessed=preprocess_bounding_box(template, vertices, operation="binarization_cc")
        image_preprocessed = preprocess_bounding_box(image, vertices, operation="binarization_cc")
        contours_template, _ = cv2.findContours(template_preprocessed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_image, _ = cv2.findContours(image_preprocessed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return len(contours_image) - len(contours_template)
    elif operation == "texture":
        template_preprocessed=preprocess_bounding_box(template, vertices, operation="gray")
        image_preprocessed = preprocess_bounding_box(image, vertices, operation="gray")

        # Compute GLCM for both images
        glcm1 = graycomatrix(template_preprocessed, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
        glcm2 = graycomatrix(image_preprocessed, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)

        # Extract texture properties
        props = ['contrast', 'correlation', 'energy', 'homogeneity']
        texture1 = [graycoprops(glcm1, prop)[0, 0] for prop in props]
        texture2 = [graycoprops(glcm2, prop)[0, 0] for prop in props]

        # Compute absolute differences for each property
        differences = [abs(t1 - t2) for t1, t2 in zip(texture1, texture2)]

        # Combine differences into a single similarity score
        similarity_score = sum(differences)  # Lower score indicates higher similarity

        return similarity_score
    elif operation=="edges":
        # Apply Canny edge detector
        template_preprocessed=preprocess_bounding_box(template, vertices, operation="gray")
        image_preprocessed = preprocess_bounding_box(image, vertices, operation="gray")
        edges1 = cv2.Canny(template_preprocessed, 100, 200)
        edges2 = cv2.Canny(image_preprocessed, 100, 200)
        # Compare edge images
        #difference = cv2.absdiff(edges1, edges2)
        #non_zero_count = cv2.countNonZero(difference)
        diff= np.mean(edges1) / (np.mean(edges2)+1e-10)
        return -diff
    elif operation=="template":
        template_preprocessed=preprocess_bounding_box(template, vertices, operation="gray")
        image_preprocessed = preprocess_bounding_box(image, vertices, operation="gray")
        # Perform template matching
        result = cv2.matchTemplate(image_preprocessed, template_preprocessed, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        return -max_val

    '''if file_name != "None":
        cv2.imwrite(f"analysis_steps\\template_{file_name}", template_preprocessed*255)
        cv2.imwrite(f"analysis_steps\\filled_{file_name}", template_preprocessed*255)'''

def show_bounding_boxes(image, bounding_boxes, file_name="image_with_bounding_boxes.png",colors=[(0,255,0),(255,0,0),(0,0,255),(255,255,0)]):
    """
    Draw bounding boxes on an image and optionally save the image.

    Parameters:
    - image (numpy array): The input image.
    - bounding_boxes (list): A list of bounding boxes, where each bounding box 
                             is a tuple (vertices, label) containing:
                             - vertices (list): List of vertices [(x1, y1), (x2, y2), ...]
                             - label (str): The label associated with the bounding box.
    - file_name (str): The name of the file to save the image with bounding boxes.
    """
    # Convert the image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Draw bounding boxes on the image
    for vertices, label, value in bounding_boxes:
        if label=="typed":
            color=colors[0]
        if label=="handwritten":
            color=colors[1]
        if label=="to_determine":
            color=colors[2]
        if label=="filled_checkbox":
            color=colors[3]
        pts = np.array(vertices, dtype=np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(image_rgb, [pts], isClosed=True, color=color, thickness=3)
        cv2.putText(image_rgb, str(round(value, 3)), (vertices[3][0], vertices[3][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0,0), 1)
        #cv2.putText(image_rgb, label[:3], (vertices[3][0], vertices[3][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0,0), 1)
    # Convert the image back to BGR for saving
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    
    # Save the image with bounding boxes if a file name is provided
    if file_name != "None":
        cv2.imwrite(f"analysis_steps\\{file_name}", image_bgr)
        print(f"Image with bounding boxes saved as {file_name}")

def filter_typed(aligned_filled_template,template,bounding_boxes,operation="cc",operation_2=None):
    result=[]
    thresholds={"average":1000,"cc":0,"ssim":-0.3,"histogram":-0.5,"orb":1,"entropy":-0.8,"contours":0,"texture":100,"edges":-0.2,"template":-0.2}
    for i,b in enumerate(bounding_boxes):
        diff=process_bounding_box(aligned_filled_template,template,b[0],operation=operation,index=i)
        diff_2=process_bounding_box(aligned_filled_template,template,b[0],operation=operation_2)
        if b[1]!="filled_checkbox":
            if operation_2 is None:
                if (diff>thresholds[operation]):
                    bounding_boxes[i][1]="handwritten" 
                else:
                    bounding_boxes[i][1]="typed"
            else:
                if (diff>thresholds[operation]):
                    #bounding_boxes[i][1]="handwritten" 
                    if(diff_2>thresholds[operation_2]):
                        bounding_boxes[i][1]="handwritten"  
                    else:
                        bounding_boxes[i][1]="typed"
                else:
                    bounding_boxes[i][1]="typed"
        #print(bounding_boxes[i][1],diff,"\n")
        result.append([b[0],bounding_boxes[i][1],i])
    return result




