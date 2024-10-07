# Jenkins Setup and Usage

## Installation

```bash
start-jenkins-apt() {
   wget -q -O - https://pkg.jenkins.io/debian-stable/jenkins.io.key | sudo apt-key add -
   sudo sh -c 'echo deb https://pkg.jenkins.io/debian-stable binary/ > /etc/apt/sources.list.d/jenkins.list'
   sudo apt update
   sudo apt install jenkins
   sudo systemctl start jenkins
   echo "Access Jenkins at http://localhost:8080"
}

start-jenkins-dnf() {
   sudo wget -O /etc/yum.repos.d/jenkins.repo https://pkg.jenkins.io/redhat-stable/jenkins.repo
   sudo rpm --import https://pkg.jenkins.io/redhat-stable/jenkins.io-2023.key
   sudo dnf upgrade -y
   sudo dnf install -y java-11-openjdk jenkins
   sudo systemctl start jenkins
   sudo systemctl enable jenkins
   sudo cat /var/lib/jenkins/secrets/initialAdminPassword
   java -jar jenkins-cli.jar -s http://localhost:8080/ -auth admin:password
}
```

## Port
``` bash
# sudo EDITOR=emacs systemctl edit jenkins
# 3. Add these lines in the editor:

# [Service]
# Environment="JENKINS_PORT=5555"
# ExecStart=
# ExecStart=/usr/bin/java -Djava.awt.headless=true -jar /usr/share/java/jenkins.war --httpPort=${JENKINS_PORT}

sudo systemctl daemon-reload
sudo systemctl restart jenkins
sudo lsof -i :5555
# http://127.0.0.1:5555
```

## Initial setup on the browser

``` bash
# Password
sudo cat /var/lib/jenkins/secrets/initialAdminPassword
```

To run a job based on a Jenkinsfile:

1. In Jenkins, click "New Item"
2. Enter a name for your job and select "Pipeline"
3. In the job configuration, scroll to the "Pipeline" section
4. Choose "Pipeline script from SCM" in the "Definition" dropdown
5. Select your SCM (e.g., Git)
6. Enter your repository URL
7. Specify the branch containing your Jenkinsfile
8. Set the "Script Path" to the location of your Jenkinsfile (default: "Jenkinsfile")
in my case, `.jenkins/Jenkinsfile`??
9. Save the configuration
10. Run the job by clicking "Build Now"

Jenkins will fetch the Jenkinsfile from your repository and execute the defined pipeline.

## Structure
project-root/
├── .jenkins/
│   ├── Jenkinsfile
│   └── config/
│       └── checkstyle.xml
├── scripts/
│   ├── build.sh
│   └── test.sh
└── README.md

## Jenkinsfile

``` groovy
pipeline {
    agent any

    stages {
        stage('Run Scripts') {
            steps {
                sh './scripts/load/all.sh'
                sh './scripts/demographic/all.sh'
                // sh './scripts/ripple/all.sh'
                // sh './scripts/GPFA/all.sh'
                // sh './scripts/NT/all.sh'
                // sh './scripts/memory_load/all.sh'
            }
        }
}
```
