import argparse
from utality import predict
import tensorflow as tf
import tensorflow_hub as hub
import json 

top_K_limit = 103

def get_input_args():
    # Create parser object
    parser = argparse.ArgumentParser(description='Process input arguments for flower prediction.')

    # Positional arguments
    parser.add_argument('image_path', type=str, help='Path to the image file.')
    parser.add_argument('model_path', type=str, help='Path to the saved model file.')

    # Optional arguments
    parser.add_argument('--top_k', type=int, default=5, help='Return the top K most likely classes.')
    parser.add_argument('--category_names', type=str, help='Path to a JSON file mapping labels to flower names.')

    # Parse the arguments
    return parser.parse_args()


def get_category_names(category_names_path):
    with open(category_names_path, 'r') as f:
        category_names = json.load(f)
    class_name = []
    for i in range(len(classes)):
        class_name.append(category_names.get(str(classes[i]), str(classes[i])))
    return class_name


if __name__ == "__main__":
    args = get_input_args()
    # uset_input = input("Enter:")

    # You can now access the arguments using `args` object
    print("Image Path:", args.image_path)
    print("Model Path:", args.model_path)
    print("Top K:", args.top_k)
    print("Category Names Path:", args.category_names)

    if args.top_k < top_K_limit and args.top_k > 0:
        keras_model = tf.keras.models.load_model(args.model_path, custom_objects={'KerasLayer': hub.KerasLayer})

        probs, classes = predict(args.image_path, keras_model, args.top_k)
        
        print("\nTop Predictions:")
        
        if args.category_names:
            class_names = get_category_names(args.category_names)
            for i in range(len(class_names)):
                print(f"Class: {class_names[i]}, Probability: {probs[i]:.4f}")
        else:
            for i in range(len(classes)):
                print(f"Class: {classes[i]}, Probability: {probs[i]:.4f}")

    else: 
        print('K must be between 1 and 102.')