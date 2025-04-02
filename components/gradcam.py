import cv2
import sys
import numpy as np

import matplotlib
matplotlib.use('Agg')  # Force non-GUI backend

import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import Model

from utils import logging, CustomException

class GradCAMPP:
    def __init__(self, model, layer_name="attention_gate", **kwargs):
        self.model = model
        self.layer_name = layer_name
        logging.info(f"Initialized GradCAMPP with layer: {self.layer_name}")

    def heatmap(self, img, category_id=None):
        try:
            img_tensor = img if img.shape[0] == 1 else np.expand_dims(img, axis=0)
            logging.info("Image tensor prepared for heatmap computation.")

            conv_layer = self.model.get_layer(self.layer_name)
            heatmap_model = Model([self.model.inputs], [conv_layer.output, self.model.output])

            with tf.GradientTape() as gtape1:
                with tf.GradientTape() as gtape2:
                    with tf.GradientTape() as gtape3:
                        conv_output, predictions = heatmap_model(img_tensor)
                        if category_id is None:
                            category_id = np.argmax(predictions[0])
                        output = predictions[:, category_id]
                        conv_first_grad = gtape3.gradient(output, conv_output)
                    conv_second_grad = gtape2.gradient(conv_first_grad, conv_output)
                conv_third_grad = gtape1.gradient(conv_second_grad, conv_output)

            global_sum = np.sum(conv_output, axis=(0, 1, 2))

            alpha_num = conv_second_grad[0]
            alpha_denom = conv_second_grad[0]*2.0 + conv_third_grad[0]*global_sum
            alpha_denom = np.where(alpha_denom != 0.0, alpha_denom, 1e-10)

            alphas = alpha_num/alpha_denom
            alpha_normalization_constant = np.sum(alphas, axis=(0,1))
            alphas /= alpha_normalization_constant

            weights = np.maximum(conv_first_grad[0], 0.0)

            deep_linearization_weights = np.sum(weights*alphas, axis=(0,1))
            grad_cam_map = np.sum(deep_linearization_weights*conv_output[0], axis=2)

            heatmap = np.maximum(grad_cam_map, 0)
            max_heat = np.max(heatmap)
            if max_heat == 0:
                max_heat = 1e-10
            heatmap /= max_heat

            logging.info("Heatmap computation completed successfully.")
            return heatmap

        except Exception as e:
            logging.error(f"Error during heatmap computation: {e}")
            raise CustomException(e, sys)

    def show(self, img, heatmap, alpha=0.4, save_path=None, return_array=False):
        try:
            # Resize the heatmap to match the original image dimensions
            heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
            logging.info("Heatmap resized to match image dimensions.")

            # Apply color map to the heatmap
            heatmap = (heatmap * 255).astype("uint8")
            heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

            # Create superimposed image
            superimposed_img = heatmap_colored * alpha + img
            superimposed_img = np.clip(superimposed_img, 0, 255).astype("uint8")

            # Create the combined plot
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))

            # Remove space around subplots
            fig.subplots_adjust(wspace=0, hspace=0)

            # Original Image
            axes[0].imshow(img)
            axes[0].set_title('Original Image')
            axes[0].axis('off')

            # Heatmap
            axes[1].imshow(heatmap_colored)
            axes[1].set_title('Heatmap')
            axes[1].axis('off')

            # Superimposed Image
            axes[2].imshow(superimposed_img)
            axes[2].set_title('Superimposed Image')
            axes[2].axis('off')

            # Adjust layout
            plt.tight_layout()

            # Save the figure if save_path is provided
            if save_path:
                fig.savefig(save_path, dpi=1200)
                logging.info(f"Saved combined visualization to {save_path}")

            plt.close(fig)  # Close the figure to avoid GUI issues

            # Return superimposed image if return_array is True
            if return_array:
                logging.info("Returning superimposed image array.")
                return superimposed_img

        except Exception as e:
            logging.error(f"Error during visualization: {e}")
            raise CustomException(e, sys)