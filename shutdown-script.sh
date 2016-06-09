# move all files over to deepaccent-results
INSTANCE=$(hostname)
sudo rm -r /model/.git
sudo mv /model /mnt/deepaccent-results/$INSTANCE/model

#shutdown and delete the instance
sudo gcloud compute instances delete $INSTANCE --zone us-central1-f -q

