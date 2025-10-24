# Mini-XDR Azure Infrastructure
# Variable Definitions

variable "project_name" {
  description = "Project name used for resource naming"
  type        = string
  default     = "mini-xdr"
}

variable "environment" {
  description = "Environment (dev, staging, prod)"
  type        = string
  default     = "prod"
}

variable "location" {
  description = "Azure region for resources"
  type        = string
  default     = "eastus"
}

variable "resource_group_name" {
  description = "Resource group name"
  type        = string
  default     = "mini-xdr-prod-rg"
}

variable "vnet_address_space" {
  description = "Virtual network address space"
  type        = list(string)
  default     = ["10.0.0.0/16"]
}

variable "aks_subnet_prefix" {
  description = "AKS subnet address prefix"
  type        = string
  default     = "10.0.4.0/22"  # Larger subnet for AKS IP requirements (1024 IPs, 10.0.4.0-10.0.7.255)
}

variable "services_subnet_prefix" {
  description = "Services subnet address prefix"
  type        = string
  default     = "10.0.2.0/24"
}

variable "appgw_subnet_prefix" {
  description = "Application Gateway subnet address prefix"
  type        = string
  default     = "10.0.3.0/24"
}

variable "corp_network_subnet_prefix" {
  description = "Mini corporate network subnet address prefix"
  type        = string
  default     = "10.0.10.0/24"
}

variable "agents_subnet_prefix" {
  description = "Agents subnet address prefix"
  type        = string
  default     = "10.0.20.0/24"
}

variable "aks_node_count" {
  description = "Number of nodes in AKS cluster"
  type        = number
  default     = 1  # Single node to fit within 10-core quota
}

variable "aks_node_size" {
  description = "VM size for AKS nodes"
  type        = string
  default     = "Standard_D2s_v3"  # 2 vCPUs, 8 GB RAM (quota-friendly)
}

variable "postgres_admin_username" {
  description = "PostgreSQL administrator username"
  type        = string
  default     = "minixdradmin"
}

variable "domain_controller_size" {
  description = "VM size for domain controller"
  type        = string
  default     = "Standard_D2s_v3"
}

variable "windows_endpoint_size" {
  description = "VM size for Windows endpoints"
  type        = string
  default     = "Standard_B2s"  # 2 vCPUs, 4 GB RAM (burstable, quota-friendly)
}

variable "windows_endpoint_count" {
  description = "Number of Windows endpoint VMs"
  type        = number
  default     = 1  # Reduced to fit within CPU quota
}

variable "linux_server_size" {
  description = "VM size for Linux servers"
  type        = string
  default     = "Standard_B2s"
}

variable "linux_server_count" {
  description = "Number of Linux server VMs"
  type        = number
  default     = 1  # Reduced to 1 to stay within 10-core vCPU quota
}

variable "admin_username" {
  description = "Admin username for VMs"
  type        = string
  default     = "azureadmin"
}

variable "enable_bastion" {
  description = "Enable Azure Bastion for secure VM access"
  type        = bool
  default     = true
}

variable "enable_postgres" {
  description = "Enable Azure PostgreSQL Flexible Server"
  type        = bool
  default     = true
}

variable "enable_redis" {
  description = "Enable Azure Cache for Redis"
  type        = bool
  default     = true
}

variable "enable_mini_corp_network" {
  description = "Deploy mini corporate network VMs"
  type        = bool
  default     = true
}

variable "tags" {
  description = "Tags to apply to all resources"
  type        = map(string)
  default = {
    Project     = "Mini-XDR"
    Environment = "Production"
    ManagedBy   = "Terraform"
  }
}

