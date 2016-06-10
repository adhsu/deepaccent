# adhsu.sh

# set -v
username=$1


# Talk to the metadata server to get the project id
# PROJECTID=$(curl -s "http://metadata.google.internal/computeMetadata/v1/project/project-id" -H "Metadata-Flavor: Google")

# Install logging monitor. The monitor will automatically pickup logs sent to
# syslog.
# curl -s "https://storage.googleapis.com/signals-agents/logging/google-fluentd-install.sh" | bash
# service google-fluentd restart &

# Install dependencies from apt
# apt-get update
# apt-get install -yq \
#     git build-essential supervisor python python-dev python-pip libffi-dev \
#     libssl-dev



mkdir /home/$username/mnt
mkdir /home/$username/mnt/deepaccent-results
chmod a+w /home/$username/mnt/deepaccent-results
gcsfuse deepaccent-results /home/$username/mnt/deepaccent-results 
mkdir /home/$username/mnt/deepaccent-data
chmod a+w /home/$username/mnt/deepaccent-data
gcsfuse deepaccent-data /home/$username/mnt/deepaccent-data



# Make sure the pythonapp user owns the application code

# Configure supervisor to start gunicorn inside of our virtualenv and run the
# applicaiton.


# Application should now be running under supervisor
