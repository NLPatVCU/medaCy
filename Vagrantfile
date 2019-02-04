# -*- mode: ruby -*-
# vi: set ft=ruby :

Vagrant.configure("2") do |config|
  config.vm.box = "ubuntu/bionic64"

config.vm.provider "virtualbox" do |v|
  v.name = "medaCy_box"
  v.gui = false
  v.memory = 3000
  v.cpus = 2
end

config.vm.provision "shell", inline: <<-SHELL
    apt-get update
    apt-get install -y python3 python3-pip python3-dev
    python3 -m pip install --upgrade pip
    pip3 install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.0.0/en_core_web_sm-2.0.0.tar.gz#egg=en_core_web_sm-2.0.0
    pip3 install -e /vagrant
SHELL
end