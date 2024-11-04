# Robotic_Surgery_Report_Generation
**Abstract :**

With the rise of robotic surgeries, automating the task of writing surgical reports has become a valuable advancement in the medical field. This project introduces a deep learning-based model designed to generate surgical reports, utilizing graph representations to capture relationships within surgical procedures for enhanced comprehension and processing. Our model incorporates a Graph Neural Network (GNN) to structure these relational data and a BERT-based language model to generate coherent, contextually accurate text. The combination of graph representation and BERT enables the system to produce high-quality reports, ultimately streamlining documentation and allowing medical professionals to focus on patient care.

**To train the model :**

Download the data and captions and model weights from [https://kennesawedu-my.sharepoint.com/personal/czhao4_kennesaw_edu/_layouts/15/onedrive.aspx?e=5%3Acd7ff596cdf04d7fbfbde02967a0c850&sharingv2=true&fromShare=true&at=9&CID=0eb75736%2D2355%2D4fef%2D8515%2Dd3a396c6e641&id=%2Fpersonal%2Fczhao4%5Fkennesaw%5Fedu%2FDocuments%2FResearch%2FAkshay&FolderCTID=0x012000351DFEEC48B9C248BBA0D2BE4531A027&view=0]

To run the model with the greedy search:

python greedy.py --trains_caption annotations_resnet/captions_train.json --validation_caption annotations_resnet/captions_val.json --image_extractions Robotic_segmentation

To run the model with beam search:

python beam.py --trains_caption annotations_resnet/captions_train.json --validation_caption annotations_resnet/captions_val.json --image_extractions Robotic_segmentation

**To validate the model :**

For greedy search:-

python validate_greedy.py  --validation_caption annotations_resnet/captions_val.json --image_extractions Robotic_segmentation –-model_weights nobeam.pth

For beam search:-

python validate_beam.py  --validation_caption annotations_resnet/captions_val.json --image_extractions Robotic_segmentation –-model_weights beam.pth







