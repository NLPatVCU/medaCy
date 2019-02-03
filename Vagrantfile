# -*- mode: ruby -*-
# vi: set ft=ruby :

Vagrant.configure("2") do |config|
  config.vm.box = "ubuntu/bionic64"

config.vm.provider "virtualbox" do |v|
  v.name = "medaCy_box"
  end

config.vm.provider "virtualbox" do |v|
  v.memory = 3000
  v.cpus = 2
end

config.vm.provision "shell", inline: <<-SHELL
    apt-get update
    apt-get install -y python3 python3-pip python3-dev
    python3 -m pip install --upgrade pip
    pip3 install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.0.0/en_core_web_sm-2.0.0.tar.gz#egg=en_core_web_sm-2.0.0
    sudo -H pip3 install -e /vagrant

    # Optional peripheral packages
    sudo -H pip3 install git+https://github.com/NanoNLP/medaCy_dataset_end.git
SHELL
end