# Spec

Before the changes, the model was trained on just 10 epoch and with vanilla Cross Entropy function without data augmentation, so here is the change:

- Data augmentation
- 50 epochs
- Use Focal Cross Entropy Loss Function with background class being 10% weight

# Result:

| Model                            | Pixel Accuracy | Mean IoU |
| -------------------------------- | -------------- | -------- |
| TransUNet (50 epoch, focal loss) | 59.80%         | 8.63%    |
| U-Net (50 epoch, focal loss)     | 60.82%         | 9.86%    |
