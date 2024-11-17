import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Load your CNN model (replace 'model_path' with the path to your .keras model file)
model = tf.keras.models.load_model('/_Projects/Shapenet/app/02691156.keras')

# Function to preprocess image for model prediction
def preprocess_image(image: Image.Image):
    # Resize image to (128, 128) and convert to grayscale
    image = image.convert("L")  # Convert to grayscale
    image = image.resize((128, 128))  # Resize to match input shape of the model
    image_array = np.array(image)  # Convert to numpy array
    image_array = image_array.astype('float32') / 255.0  # Normalize image
    image_array = np.expand_dims(image_array, axis=-1)  # Add channel dimension (128, 128, 1)
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension (1, 128, 128, 1)
    return image_array

# Function to save 3D object as OBJ file
def save_obj_file(output, filename='output.obj'):

    predicted_voxel = output[0, :, :, :, 0]

    # Convert the voxel grid to a binary format (can modify threshold)
    predicted_voxel_binary = (predicted_voxel > 0.5).astype(np.uint8)

    # Vertices and faces for an individual cube (each voxel)
    vertex_offsets = [
        (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
        (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)
    ]
    
    face_offsets = [
        (0, 1, 2, 3), # Bottom
        (4, 5, 6, 7), # Top
        (0, 1, 5, 4), # Front
        (2, 3, 7, 6), # Back
        (1, 2, 6, 5), # Right
        (3, 0, 4, 7)  # Left
    ]

    vertices = []
    faces = []
    vertex_index = 1

    # Iterate through the 3D array and process each voxel
    for x in range(predicted_voxel_binary.shape[0]):
        for y in range(predicted_voxel_binary.shape[1]):
            for z in range(predicted_voxel_binary.shape[2]):
                if predicted_voxel_binary[x, y, z] == 1:  # Only process occupied voxels (value 1)
                    # Add vertices for this voxel's cube
                    for dx, dy, dz in vertex_offsets:
                        vertices.append((x + dx, y + dy, z + dz))

                    # Add faces using the latest 8 vertices
                    for face in face_offsets:
                        faces.append((
                            vertex_index + face[0],
                            vertex_index + face[1],
                            vertex_index + face[2],
                            vertex_index + face[3]
                        ))

                    # Update the vertex index
                    vertex_index += 8

    # Write vertices and faces to an OBJ file
    with open(filename, 'w') as f:
        for vertex in vertices:
            f.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
        for face in faces:
            f.write(f"f {face[0]} {face[1]} {face[2]} {face[3]}\n")

    return filename

# Streamlit UI
st.title("2D Image to 3D Object Prediction")
st.write("Upload a PNG or JPG image, and get a 3D OBJ file as output.")

# File upload
uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg"])

if uploaded_file is not None:
    # Read image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess image for the model
    image_array = preprocess_image(image)

    # Model prediction
    with st.spinner("Predicting 3D object..."):
        predicted_output = model.predict(image_array)
    
    # Save the prediction as OBJ file
    obj_filename = save_obj_file(predicted_output, 'predicted_output.obj')
    
    # Provide download link
    st.success("Prediction complete! You can download the 3D object below.")
    with open(obj_filename, "rb") as f:
        st.download_button(
            label="Download 3D OBJ file",
            data=f,
            file_name=obj_filename,
            mime="application/octet-stream"
        )
