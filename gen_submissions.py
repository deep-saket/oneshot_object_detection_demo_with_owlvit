from demo_image_conditioned import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser("OWL-ViT one shot object detection", add_help=True)

    parser.add_argument("--shelf_dir", "-i", type=str, required=True, help="path to image file")
    parser.add_argument("--query_dir", "-qi", type=str, default="", required=True, help="path to query image file")
    
    parser.add_argument(
        "--output_dir", "-o", type=str, default="outputs", required=True, help="output directory"
    )
    parser.add_argument('--owlvit_model', help='select model', default="owlvit-base-patch32", choices=["owlvit-base-patch32", "owlvit-base-patch16", "owlvit-large-patch14"])
    parser.add_argument("--box_threshold", type=float, default=0.0, help="box threshold")
    parser.add_argument("--nms_threshold", type=float, default=0.0, help="nms threshold")
    parser.add_argument('--device', help='select device', default="cuda:0", type=str)
    args = parser.parse_args()

    # load arguments
    shelf_dir = args.shelf_dir
    query_dir = args.query_dir
    output_dir = args.output_dir
    box_threshold = args.box_threshold
    nms_threshold = args.nms_threshold


    # create dir to store outputs
    os.makedirs(output_dir, exist_ok=True)

    # load OWL-ViT model
    model, processor = load_owlvit(checkpoint_path=args.owlvit_model, device=args.device)

    for query_image_name in os.listdir(query_dir):
        query_image_path = os.path.join(query_dir, query_image_name)
        query_image_class = query_image_name[2:-4]

        for shelf_image_name in os.listdir(shelf_dir):
            shelf_image_path = os.path.join(shelf_dir, shelf_image_name)            
            shelf_image_class = shelf_image_name[2:-4]

            print(query_image_path, shelf_image_path)

            # load image
            image = Image.open(shelf_image_path)    

            # run object detection model
            with torch.no_grad():
                query_image = Image.open(query_image_path).convert('RGB')
                inputs = processor(query_images=query_image, images=image, return_tensors="pt").to(args.device)
                outputs = model.image_guided_detection(**inputs)
    
            # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
            target_sizes = torch.Tensor([image.size[::-1]])
            # Convert outputs (bounding boxes and class logits) to COCO API
            results = processor.post_process_image_guided_detection(outputs=outputs, threshold=box_threshold, nms_threshold=nms_threshold, target_sizes=target_sizes.to(args.device))
            scores = torch.sigmoid(outputs.logits)
            topk_scores, topk_idxs = torch.topk(scores, k=1, dim=1)
            
            i = 0  # Retrieve predictions for the first image
            
            boxes, scores = results[i]["boxes"], results[i]["scores"]
            

            # Print detected objects and rescaled box coordinates
            for box, score in zip(boxes, scores):
                box = [round(i, 2) for i in box.tolist()]
                print(f"Detected object with confidence {round(score.item(), 3)} at location {box}")

            boxes = boxes.cpu().detach().numpy()
            normalized_boxes = copy.deepcopy(boxes)
            
            # # visualize pred
            size = image.size   

            with open('a.txt', 'w') as f:
                for box in normalized_boxes:
                    f.write(f'{query_image_class} {shelf_image_class} {int(box[0])} {int(box[1])} {int(box[2])} {int(box[3])}\n')