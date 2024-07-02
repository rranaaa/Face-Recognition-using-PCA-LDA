def process():
    import numpy as np
    from PIL import Image
    #import matplotlib.pyplot as plt
    
    # Define the image size (92 x 112)
    image_size = (92, 112)
    
    # Define the number of subjects and images per subject
    num_subjects = 40
    images_per_subject = 10
    
    # Initialize the Data Matrix and Label vector
    '''The Data Matrix D:
    is a two-dimensional array that 
    holds the pixel values of all images in the dataset.
    '''
    D = np.zeros((num_subjects * images_per_subject, image_size[0] * image_size[1])) # row=400 , col=10304
    y = np.zeros(num_subjects * images_per_subject) # labels 400
    
    
    # Process each subject's images
    for subject in range(1, num_subjects + 1):
        for num in range(1, images_per_subject + 1):
            # Load the image
            image_path = f"./Data/s{subject}/{num}.pgm"
            
            '''Why we convert it to grayscale?
            Dimensionality Reduction: Color images typically have three channels (red, green, and blue),
                                        whereas grayscale images have only one channel representing the intensity of light.
                                        By converting to grayscale, you reduce the dimensionality of each image from three channels to one,
                                        simplifying the data representation. This can make processing more efficient, 
                                        especially for algorithms that are not designed to handle color images.
            Standardization: Converting images to grayscale ensures that all images have the same number of channels,
                                which can be important for consistency in processing and analysis.
                                It removes any potential variations due to color that may not be relevant for the specific task at hand
            '''
            image = Image.open(image_path).convert("L")  # Convert to grayscale
            
           
            '''.flatten()?        
            The .flatten() method in NumPy is used to transform a multi-dimensional array into
            one-dimensional array by collapsing all of its dimensions.
            '''
            image_vector = np.array(image).flatten() # Convert the image to a vector
            
            
            ''' compute index
            index: if subject = 1  & image num = 1 so index = 0
            and son on:if subject = 40 & image num = 10 so index = 399
            so data matrix contain 400 image or row from 0 to 399
            '''
            index = (subject - 1) * images_per_subject + num - 1 
            D[index] = image_vector # Add the image vector to the Data Matrix
            
            # Set the label for the image
            y[index] = subject
            
             #Display the 400 image! (To see it go to plots)
            #plt.imshow(image, cmap="gray")
            #plt.title(f"Subject: {subject}, Image: {num}")
            #plt.axis("off")
            #plt.show()
    
    # Print the shapes of the Data Matrix and Label vector
    '''Data Matrix shape:
    The shape (400, 10304) means that there are :
    400 rows and 10304 columns in the Data Matrix.
    Each row represents an image, and each column represents a pixel in the image.
    So, there are a total of 400 images, and each image has 10304 pixels which
    means image size (92 x 112).
    '''
    print("Label vector shape:", y.shape)
    print("Data Matrix shape:", D.shape)
    print("Data Matrix shape::", D.shape)
    
    '''Label vector shape:
    1-dim. every 10 elemnts contains the number of image ascendingly
    '''
    print("Label vector shape:", y.shape)
    
    # Split the dataset into training and test sets
    X_train = D[::2]  # Select odd rows for training 0,2,4...
    y_train = y[::2]
    
    X_test = D[1::2]  # Select even rows for testing 1,3,5...
    y_test = y[1::2]
    
    # Print the shapes of the training and test sets
    print("\nTraining set shapes:")
    print("X_train shape:", X_train.shape) #200 row with the same 10304 columns
    print("y_train shape:", y_train.shape) # 200 elemnts
    
    print("\nTest set shapes:")
    print("X_test shape:", X_test.shape) #200 row with the same 10304 columns
    print("y_test shape:", y_test.shape) #200 elements
    
    return X_train, y_train, X_test, y_test