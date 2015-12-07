The Code is completely written in python 

Dependencies:
1. numpy
2. scipy
3. sklearn

Setting the running environment:
1. install the pip to install the above dependencies
2. if in linux - sudo apt-get install pip
         mac - brew install pip
3. following to install the above dependencies
    pip install numpy
    pip install scipy
    pip install sklearn

File structure:

mlhw-
    src
        classification.py
        missing_values.py
    results
        venky_classification*.txt
        venky_missingresult*.txt
    data
        newtest*/
            TrainData.txt
            TestData.txt
            TrainLabel.txt
        misstest*/
            MissingData.txt

How to run the code:

running classification code
    python classification.py newtest1

running missing values code
    python missing_values.py misstest1
