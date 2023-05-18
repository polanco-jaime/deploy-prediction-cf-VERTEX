# Analyzing Satellite Imagery via BigQuery SQL
The goal of these steps is to run a BigQuery SQL and extract information from a model of vertex AI

## Requirements
* Ensure your GCP user has access to VERTEX AI.

## Setting up the demo
**1)** In Cloud Shell or other environment where you have the gcloud SDK installed, execute the following commands:
```console
gcloud components update 
cd $HOME
git clone https://github.com/polanco-jaime/earth-engine-on-bigquery.git
cd ~/earth-engine-on-bigquery
chmod +x *.sh
```

**2)** **Edit config.sh** - In your editor of choice update the variables in config.sh to reflect your desired gcp project.

**3)** Next execute the command below

```console
sh setup_sa.sh
```
If the shell script has executed successfully, you should now have a new Service Account created, as shown in the image below
<br/><br/>
![Setup output](/img/setup_sa.png)

**4)** A Service Account(SA) in format <Project_Number-compute@developer.gserviceaccount.com> was created in previous step, you need to signup this SA for Earth Engine at [EE SA signup](https://signup.earthengine.google.com/#!/service_accounts). Check out the last line of the screenshot above it will list out SA name. On successful signup, you screen should look like the one below.

![Signup SA](/img/signup_sa.png)


**5)** Next execute the command below, before that, verify you are registered in [Google Earth Enine](https://developers.google.com/earth-engine/guides/access#a-role-in-a-cloud-project)

```console
sh deploy_cf.sh
```

If the shell script has executed successfully,have a dataset gee and table land_coords under your project in BigQuery along with a functions get_poly_ndvi_month and get_poly_temp_month. 
 
## Congrats! You just executed BigQuery SQL over Landsat imagery
