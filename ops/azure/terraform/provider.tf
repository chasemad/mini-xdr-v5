# Mini-XDR Azure Infrastructure
# Provider Configuration

terraform {
  required_version = ">= 1.0"
  
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.80"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.5"
    }
  }
  
  # Backend configuration for state management
  # Uncomment after creating storage account
  # backend "azurerm" {
  #   resource_group_name  = "mini-xdr-terraform-rg"
  #   storage_account_name = "minixdrterraformstate"
  #   container_name       = "tfstate"
  #   key                  = "mini-xdr.tfstate"
  # }
}

provider "azurerm" {
  features {
    resource_group {
      prevent_deletion_if_contains_resources = false
    }
    
    key_vault {
      purge_soft_delete_on_destroy    = true
      recover_soft_deleted_key_vaults = true
    }
    
    virtual_machine {
      delete_os_disk_on_deletion     = true
      graceful_shutdown              = false
      skip_shutdown_and_force_delete = false
    }
  }
}

# Get current client configuration
data "azurerm_client_config" "current" {}

# Get user's public IP for NSG rules
data "http" "myip" {
  url = "https://ifconfig.me/ip"
}

locals {
  my_ip = chomp(data.http.myip.response_body)
}

