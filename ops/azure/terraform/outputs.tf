# Mini-XDR Azure Infrastructure
# Output Values

output "resource_group_name" {
  description = "The name of the resource group"
  value       = azurerm_resource_group.main.name
}

output "resource_group_location" {
  description = "The location of the resource group"
  value       = azurerm_resource_group.main.location
}

output "vnet_id" {
  description = "The ID of the virtual network"
  value       = azurerm_virtual_network.main.id
}

output "aks_cluster_name" {
  description = "The name of the AKS cluster"
  value       = azurerm_kubernetes_cluster.main.name
}

output "aks_cluster_id" {
  description = "The ID of the AKS cluster"
  value       = azurerm_kubernetes_cluster.main.id
}

output "aks_kube_config" {
  description = "Kubectl configuration (sensitive)"
  value       = azurerm_kubernetes_cluster.main.kube_config_raw
  sensitive   = true
}

output "acr_login_server" {
  description = "The login server of the Azure Container Registry"
  value       = azurerm_container_registry.main.login_server
}

output "acr_id" {
  description = "The ID of the Azure Container Registry"
  value       = azurerm_container_registry.main.id
}

output "postgres_fqdn" {
  description = "The FQDN of the PostgreSQL server"
  value       = var.enable_postgres ? azurerm_postgresql_flexible_server.main[0].fqdn : null
}

output "postgres_database_name" {
  description = "The name of the PostgreSQL database"
  value       = var.enable_postgres ? azurerm_postgresql_flexible_server_database.minixdr[0].name : null
}

output "redis_hostname" {
  description = "The hostname of the Redis cache"
  value       = var.enable_redis ? azurerm_redis_cache.main[0].hostname : null
}

output "key_vault_name" {
  description = "The name of the Key Vault"
  value       = azurerm_key_vault.main.name
}

output "key_vault_uri" {
  description = "The URI of the Key Vault"
  value       = azurerm_key_vault.main.vault_uri
}

output "appgw_public_ip" {
  description = "The public IP address of the Application Gateway"
  value       = azurerm_public_ip.appgw.ip_address
}

output "bastion_public_ip" {
  description = "The public IP address of the Bastion host"
  value       = var.enable_bastion ? azurerm_public_ip.bastion[0].ip_address : null
}

output "domain_controller_private_ip" {
  description = "The private IP address of the domain controller"
  value       = var.enable_mini_corp_network ? azurerm_network_interface.dc[0].private_ip_address : null
}

output "windows_endpoint_private_ips" {
  description = "The private IP addresses of Windows endpoints"
  value       = var.enable_mini_corp_network ? [for nic in azurerm_network_interface.endpoints : nic.private_ip_address] : []
}

output "linux_server_private_ips" {
  description = "The private IP addresses of Linux servers"
  value       = var.enable_mini_corp_network ? [for nic in azurerm_network_interface.servers : nic.private_ip_address] : []
}

output "your_ip_address" {
  description = "Your detected IP address (used for NSG rules)"
  value       = local.my_ip
}

output "vm_admin_username" {
  description = "The admin username for VMs"
  value       = var.admin_username
}

output "log_analytics_workspace_id" {
  description = "The ID of the Log Analytics workspace"
  value       = azurerm_log_analytics_workspace.main.id
}

