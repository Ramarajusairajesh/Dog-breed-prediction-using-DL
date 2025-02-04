For this project based on Convolutional Neural Networks (CNNs) to detect the breeds of different dogs. For the dataset, I used a smaller version of the Stanford Dogs dataset and split it into two parts: 70% for training and 30% for testing.

For the models, I chose **DenseNet**, **Inception**, and **Xception**, training them on the images. To improve accuracy, I implemented a **Majority Voting algorithm**, where the three models
continuously predict the breed of a dog. The process continues until at least two out of the three models agree on a breed, significantly reducing the final error.

I selected **DenseNet**, **Inception**, and **Xception** because they each bring unique advantages to image classification tasks, especially in fine-grained recognition like dog breed
identification. DenseNet improves feature reuse and reduces the vanishing gradient problem by densely connecting layers, making it efficient for smaller datasets.
Inception captures multi-scale features using parallel convolutional filters of different sizes, making it effective for distinguishing subtle differences between breeds. 
Xception, an advanced variant of Inception, leverages depthwise separable convolutions to enhance spatial feature extraction while reducing computational cost, improving both 
accuracy and efficiency.  

To achieve better classification results, I implemented a Majority Voting algorithm, where the three models continuously predict a dog's breed until at least two of them agree on the
same label. This ensemble approach helps minimize errors caused by individual model weaknesses and improves overall robustness. By combining the strengths of different architectures,
the final system reduces misclassification and enhances prediction confidence, making it more reliable for real-world applications.
