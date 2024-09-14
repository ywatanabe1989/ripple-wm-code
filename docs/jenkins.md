
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

## Structure
project-root/
├── .jenkins/
│   ├── Jenkinsfile
│   ├── scripts/
│   │   ├── build.sh
│   │   └── test.sh
│   └── config/
│       └── checkstyle.xml
├── src/
├── tests/
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
                sh './scripts/ripple/all.sh'
                sh './scripts/GPFA/all.sh'
                sh './scripts/NT/all.sh'
                sh './scripts/memory_load/all.sh'
            }
        }
        stage('Run Tests') {
            steps {
                sh 'python -m unittest discover tests'
            }
        }
        stage('Lint') {
            steps {
                sh 'pylint **/*.py'
            }
        }
    }
}
```
