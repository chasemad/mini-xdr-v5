# Mini-XDR Azure Infrastructure
# Database Services Configuration

# PostgreSQL Flexible Server
resource "azurerm_postgresql_flexible_server" "main" {
  count               = var.enable_postgres ? 1 : 0
  name                = "${var.project_name}-postgres"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  version             = "14"
  
  administrator_login    = var.postgres_admin_username
  administrator_password = random_password.postgres.result
  
  sku_name   = "GP_Standard_D2s_v3"
  storage_mb = 131072  # 128 GB
  zone       = "1"     # Explicitly set zone to prevent modification errors

  backup_retention_days        = 7
  geo_redundant_backup_enabled = false
  
  delegated_subnet_id = azurerm_subnet.services.id
  private_dns_zone_id = azurerm_private_dns_zone.postgres[0].id
  
  public_network_access_enabled = false
  
  # Zone-Redundant HA not available in all regions
  # Disabled to reduce costs and ensure compatibility
  # high_availability {
  #   mode = "ZoneRedundant"
  # }
  
  maintenance_window {
    day_of_week  = 0
    start_hour   = 3
    start_minute = 0
  }
  
  tags = var.tags
  
  depends_on = [
    azurerm_private_dns_zone_virtual_network_link.postgres
  ]
}

# Private DNS Zone for PostgreSQL
resource "azurerm_private_dns_zone" "postgres" {
  count               = var.enable_postgres ? 1 : 0
  name                = "privatelink.postgres.database.azure.com"
  resource_group_name = azurerm_resource_group.main.name
  tags                = var.tags
}

resource "azurerm_private_dns_zone_virtual_network_link" "postgres" {
  count                 = var.enable_postgres ? 1 : 0
  name                  = "${var.project_name}-postgres-dns-link"
  resource_group_name   = azurerm_resource_group.main.name
  private_dns_zone_name = azurerm_private_dns_zone.postgres[0].name
  virtual_network_id    = azurerm_virtual_network.main.id
  tags                  = var.tags
}

# PostgreSQL Database
resource "azurerm_postgresql_flexible_server_database" "minixdr" {
  count     = var.enable_postgres ? 1 : 0
  name      = "minixdr"
  server_id = azurerm_postgresql_flexible_server.main[0].id
  collation = "en_US.utf8"
  charset   = "utf8"
}

# PostgreSQL Firewall Rule (allow AKS subnet)
resource "azurerm_postgresql_flexible_server_firewall_rule" "aks" {
  count            = var.enable_postgres ? 1 : 0
  name             = "AllowAKSSubnet"
  server_id        = azurerm_postgresql_flexible_server.main[0].id
  start_ip_address = cidrhost(var.aks_subnet_prefix, 0)
  end_ip_address   = cidrhost(var.aks_subnet_prefix, -1)
}

# Azure Cache for Redis
resource "azurerm_redis_cache" "main" {
  count               = var.enable_redis ? 1 : 0
  name                = "${var.project_name}-redis"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  capacity            = 1
  family              = "C"
  sku_name            = "Standard"
  enable_non_ssl_port = false
  minimum_tls_version = "1.2"
  
  redis_configuration {
    enable_authentication = true
  }
  
  public_network_access_enabled = true
  
  tags = var.tags
}

# Store Redis connection string in Key Vault
resource "azurerm_key_vault_secret" "redis_connection_string" {
  count        = var.enable_redis ? 1 : 0
  name         = "redis-connection-string"
  value        = azurerm_redis_cache.main[0].primary_connection_string
  key_vault_id = azurerm_key_vault.main.id
  
  depends_on = [azurerm_key_vault_access_policy.current_user]
}

# Store PostgreSQL connection string in Key Vault
resource "azurerm_key_vault_secret" "postgres_connection_string" {
  count        = var.enable_postgres ? 1 : 0
  name         = "postgres-connection-string"
  value        = "postgresql://${var.postgres_admin_username}:${random_password.postgres.result}@${azurerm_postgresql_flexible_server.main[0].fqdn}:5432/minixdr?sslmode=require"
  key_vault_id = azurerm_key_vault.main.id
  
  depends_on = [azurerm_key_vault_access_policy.current_user]
}

