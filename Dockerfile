From python:3

ADD MSPs_model.py  /
ADD api.py /
ADD Gender_Detection_MLPs.pth /

Run pip install pandas
Run pip install numpy
Run pip install Flask-RESTful
Run pip install Flask
Run pip install torch



CMD python MSPs_model.py
CMD python api.py 

