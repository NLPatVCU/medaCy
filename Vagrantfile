# -*- mode: ruby -*-
# vi: set ft=ruby :

Vagrant.configure("2") do |config|
  config.vm.box = "ubuntu/bionic64"
  config.disksize.size = "17GB"

config.vm.provider "virtualbox" do |v|
  v.name = "medaCy_box"
  v.gui = false
  v.memory = 4500
  v.cpus = 2
end

config.vm.provision "shell", inline: <<-SHELL
    apt-get update
    apt-get install -y python3 python3-pip python3-dev
    python3 -m pip install --upgrade pip
    pip3 install -e /vagrant

    # Optional packages for testing (for those who want to develop medaCy)
    pip3 install pytest pytest-cov
SHELL
end