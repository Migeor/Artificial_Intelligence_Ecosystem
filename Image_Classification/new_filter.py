from PIL import Image, ImageFilter, ImageEnhance
import matplotlib.pyplot as plt
import os

def apply_artistic_sketch_filter(image_path, output_path="sketch_image.png"):
    try:
        # 1. Open original image
        img = Image.open(image_path)
        
        # 2. Convert to Grayscale (Black & White)
        bw_img = img.convert("L")
        
        # 3. Find the sharp edges in the image
        edges = bw_img.filter(ImageFilter.FIND_EDGES)
        
        # 4. Invert the colors so it's a dark sketch on a light background
        # (Using a clean pixel map trick to flip 0-255 values)
        inverted_edges = edges.point(lambda x: 255 - x)
        
        # 5. Jack up the contrast to make the lines look like dark pencil strokes
        enhancer = ImageEnhance.Contrast(inverted_edges)
        final_sketch = enhancer.enhance(2.5)  # Multiplies contrast by 2.5

        # Save using matplotlib without borders
        plt.imshow(final_sketch, cmap='gray')
        plt.axis('off')
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        
        print(f"Artistic sketch image saved successfully as '{output_path}'")

    except Exception as e:
        print(f"Error processing image: {e}")

if __name__ == "__main__":
    print("Artistic Sketch Filter Processor (type 'exit' to quit)\n")
    while True:
        image_path = input("Enter image filename (or 'exit' to quit): ").strip()
        if image_path.lower() == 'exit':
            print("Goodbye!")
            break
        if not os.path.isfile(image_path):
            print(f"File not found: {image_path}")
            continue
            
        base, ext = os.path.splitext(image_path)
        output_file = f"{base}_sketch{ext}"
        apply_artistic_sketch_filter(image_path, output_file)