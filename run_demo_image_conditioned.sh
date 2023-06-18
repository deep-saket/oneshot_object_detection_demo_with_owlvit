python demo_image_conditioned.py --image_path demo_images/db3103.jpg \
                                 --query_image_path demo_images/qr18.jpg \
                                 --device cpu:0 --output_dir ./op1 \
                                 --owlvit_model owlvit-base-patch16 \
                                 --box_threshold 0.66
