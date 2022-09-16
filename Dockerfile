# STEP 1: Pull python image
FROM python:3.9.13

# STEP 2,3: CREATE WORK DIR AND COPY FILE TO WORK DIR
WORKDIR /sorghum
COPY requirements.txt /sorghum

# STEP 4,5: INSTALL NECESSARY PACKAGE
RUN pip install --upgrade pip
RUN pip3 install --no-cache-dir -r requirements.txt

# STEP 6: RUN COMMAND
COPY . /sorghum
CMD ["python", "./app.py"]