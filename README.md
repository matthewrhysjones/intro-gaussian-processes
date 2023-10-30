### Introduction to Gaussian processes 

In the repo, you will find several files:

    - presentation files 
    - kernels.py
    - models.py
    - presentation-code.py
    - requirements

In the **presentation** folder, you should be able to open to main.html file to render the presentation in your browser. The presentation compiles from the .qmd file, so this contains all the python code that was ran for each slide. 

However, as .qmd files require quarto to run, I would probably not bother trying to compile it on your machine. Instead, you can go to "presentation-code.py" and run all the code from the slides. You can then match the plots here with those from the html file. 

**kernels.py** contains some custom kernel classes. Each kernel will return kxx (covariance of training set with training set), kxt (cov of train & test) and ktt (cov of test & test). Inputs are the train inputs, test inputs and hyps. All kernels take sf2, l as the first two inputs, with the observation noise as the last input. 

hyps = [sf2, l, sn2] 

Note that none of the kernels actually interact with the observation noise hyperparameter; this happens in the loss function in models.

The periodic kernel also has a hyperparameter that controls the period of the kernel draws, and so in that case the hyps would be 

hyps = [sf2, l , p, sn2]

**models.py** contains the loss, train and prediction function. The kernel should be passed to these functions as follows

kernels.SquaredExponential
</br>
kernels.Matern12

etc

You can run all the code in the presentation from **presentation-code.py**, just make sure that you have the necessary libraries installed in your environment specified in **requirements.txt**, but you can see all the packages I had installed from the first code block in *resentation-code.py. I used python 3.11.3, so can't guarantee execution on other versions, but let me know if you have any issues.  
