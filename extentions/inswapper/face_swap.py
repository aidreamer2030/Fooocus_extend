import sys
import gradio as gr
import modules.gradio_hijack as grh
from PIL import Image
import numpy as np
from modules import config
from extentions.inswapper.swapper import process
def inswapper_gui():
  with gr.Row():
    with gr.Column():
      inswapper_enabled = gr.Checkbox(label="Enabled", value=False)
      inswapper_source_image_indicies = gr.Text(label="Source Image Index", info="-1 will swap all faces, otherwise provide the 0-based index of the face (0, 1, etc)", value="0")
      inswapper_target_image_indicies = gr.Text(label = "Target Image Index", info="-1 will swap all faces, otherwise provide the 0-based index of the face (0, 1, etc)", value="0")
      inswapper_background_enhance=gr.Checkbox(label="Background Enchanced", value=True)
      inswapper_face_upsample=gr.Checkbox(label="Face Upsample", value=True)
      inswapper_upscale = gr.Slider(label='Upscale', minimum=1.0, maximum=4.0, step=1.0, value=1,interactive=True)
      inswapper_fidelity =gr.Slider(label='Codeformer_Fidelity', minimum=0, maximum=1, value=0.5, step=0.01, info='0 for better quality, 1 for better identity (default=0.5)')
    with gr.Column():
      inswapper_source_image = grh.Image(label='Source Face Image', source='upload', type='numpy')
  with gr.Row():
    gr.HTML('* \"inswapper\" is powered by haofanwang. <a href="https://github.com/haofanwang/inswapper" target="_blank">\U0001F4D4 Document</a>')
  return inswapper_enabled,inswapper_source_image_indicies,inswapper_target_image_indicies,inswapper_background_enhance,inswapper_face_upsample,inswapper_upscale,inswapper_fidelity,inswapper_source_image


def perform_face_swap(images, inswapper_source_image, inswapper_source_image_indicies, inswapper_target_image_indicies,inswapper_background_enhance,inswapper_face_upsample,inswapper_upscale,inswapper_fidelity):
  swapped_images = []

  for item in images:
      source_image = Image.fromarray(inswapper_source_image)
      print(f"Inswapper: Source indicies: {inswapper_source_image_indicies}")
      print(f"Inswapper: Target indicies: {inswapper_target_image_indicies}") 
  
      result_image = process([source_image], item, inswapper_source_image_indicies, inswapper_target_image_indicies, f"{config.path_clip_vision}/inswapper_128.onnx")

  if True:
      from restoration import face_restoration,check_ckpts,set_realesrgan,torch,ARCH_REGISTRY,cv2
      
      # make sure the ckpts downloaded successfully
      check_ckpts()
      
      # https://huggingface.co/spaces/sczhou/CodeFormer
      upsampler = set_realesrgan()
      device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
#      print(f"{device}")

      codeformer_net = ARCH_REGISTRY.get("CodeFormer")(dim_embd=512,
                                                        codebook_size=1024,
                                                        n_head=8,
                                                        n_layers=9,
                                                        connect_list=["32", "64", "128", "256"],
                                                      ).to(device)
      ckpt_path = "extentions/CodeFormer/weights/CodeFormer/codeformer.pth"
      checkpoint = torch.load(ckpt_path)["params_ema"]
      codeformer_net.load_state_dict(checkpoint)
      codeformer_net.eval()     
      result_image = cv2.cvtColor(np.array(result_image), cv2.COLOR_RGB2BGR)
      result_image = face_restoration(result_image, 
                                      inswapper_background_enhance, 
                                      inswapper_face_upsample, 
                                      inswapper_upscale, 
                                      inswapper_fidelity,
                                      upsampler,
                                      codeformer_net,
                                      device)

      swapped_images.append(result_image)
  
  return swapped_images

