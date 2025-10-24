# Mini-XDR Azure Infrastructure
# Virtual Machine Configuration for Mini Corporate Network

# Domain Controller VM
resource "azurerm_windows_virtual_machine" "dc" {
  count               = var.enable_mini_corp_network ? 1 : 0
  name                = "mini-corp-dc01"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  size                = var.domain_controller_size
  admin_username      = var.admin_username
  admin_password      = random_password.vm_admin.result
  tags                = merge(var.tags, { Role = "DomainController" })
  
  network_interface_ids = [
    azurerm_network_interface.dc[0].id
  ]
  
  os_disk {
    caching              = "ReadWrite"
    storage_account_type = "Premium_LRS"
  }
  
  source_image_reference {
    publisher = "MicrosoftWindowsServer"
    offer     = "WindowsServer"
    sku       = "2022-Datacenter"
    version   = "latest"
  }
  
  boot_diagnostics {
    storage_account_uri = azurerm_storage_account.diagnostics.primary_blob_endpoint
  }
}

resource "azurerm_network_interface" "dc" {
  count               = var.enable_mini_corp_network ? 1 : 0
  name                = "mini-corp-dc01-nic"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  tags                = var.tags
  
  ip_configuration {
    name                          = "internal"
    subnet_id                     = azurerm_subnet.corp_network.id
    private_ip_address_allocation = "Static"
    private_ip_address            = cidrhost(var.corp_network_subnet_prefix, 10)
  }
}

# Windows Endpoint VMs
resource "azurerm_windows_virtual_machine" "endpoints" {
  count               = var.enable_mini_corp_network ? var.windows_endpoint_count : 0
  name                = "mini-corp-ws${format("%02d", count.index + 1)}"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  size                = var.windows_endpoint_size
  admin_username      = var.admin_username
  admin_password      = random_password.vm_admin.result
  tags                = merge(var.tags, { Role = "WindowsEndpoint" })
  
  network_interface_ids = [
    azurerm_network_interface.endpoints[count.index].id
  ]
  
  os_disk {
    caching              = "ReadWrite"
    storage_account_type = "Standard_LRS"
  }
  
  source_image_reference {
    publisher = "MicrosoftWindowsDesktop"
    offer     = "windows-11"
    sku       = "win11-22h2-ent"
    version   = "latest"
  }
  
  boot_diagnostics {
    storage_account_uri = azurerm_storage_account.diagnostics.primary_blob_endpoint
  }
}

resource "azurerm_network_interface" "endpoints" {
  count               = var.enable_mini_corp_network ? var.windows_endpoint_count : 0
  name                = "mini-corp-ws${format("%02d", count.index + 1)}-nic"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  tags                = var.tags
  
  ip_configuration {
    name                          = "internal"
    subnet_id                     = azurerm_subnet.corp_network.id
    private_ip_address_allocation = "Dynamic"
  }
}

# Linux Server VMs
resource "azurerm_linux_virtual_machine" "servers" {
  count               = var.enable_mini_corp_network ? var.linux_server_count : 0
  name                = "mini-corp-srv${format("%02d", count.index + 1)}"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  size                = var.linux_server_size
  admin_username      = var.admin_username
  admin_password      = random_password.vm_admin.result
  tags                = merge(var.tags, { Role = "LinuxServer" })
  
  disable_password_authentication = false
  
  network_interface_ids = [
    azurerm_network_interface.servers[count.index].id
  ]
  
  os_disk {
    caching              = "ReadWrite"
    storage_account_type = "Standard_LRS"
  }
  
  source_image_reference {
    publisher = "Canonical"
    offer     = "0001-com-ubuntu-server-jammy"
    sku       = "22_04-lts"
    version   = "latest"
  }
  
  boot_diagnostics {
    storage_account_uri = azurerm_storage_account.diagnostics.primary_blob_endpoint
  }
}

resource "azurerm_network_interface" "servers" {
  count               = var.enable_mini_corp_network ? var.linux_server_count : 0
  name                = "mini-corp-srv${format("%02d", count.index + 1)}-nic"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  tags                = var.tags
  
  ip_configuration {
    name                          = "internal"
    subnet_id                     = azurerm_subnet.corp_network.id
    private_ip_address_allocation = "Dynamic"
  }
}

# Storage Account for Boot Diagnostics
resource "azurerm_storage_account" "diagnostics" {
  name                     = "${replace(var.project_name, "-", "")}diag${random_string.suffix.result}"
  resource_group_name      = azurerm_resource_group.main.name
  location                 = azurerm_resource_group.main.location
  account_tier             = "Standard"
  account_replication_type = "LRS"
  tags                     = var.tags
}

# VM Extension to configure Domain Controller
resource "azurerm_virtual_machine_extension" "dc_configure" {
  count                = var.enable_mini_corp_network ? 1 : 0
  name                 = "configure-ad-ds"
  virtual_machine_id   = azurerm_windows_virtual_machine.dc[0].id
  publisher            = "Microsoft.Compute"
  type                 = "CustomScriptExtension"
  type_handler_version = "1.10"
  
  settings = <<SETTINGS
    {
        "commandToExecute": "powershell.exe -Command \"Install-WindowsFeature -Name AD-Domain-Services -IncludeManagementTools; Write-Output 'AD-DS role installed successfully'\""
    }
  SETTINGS
  
  tags = var.tags
}

# Auto-shutdown schedule for cost savings
resource "azurerm_dev_test_global_vm_shutdown_schedule" "dc" {
  count              = var.enable_mini_corp_network ? 1 : 0
  virtual_machine_id = azurerm_windows_virtual_machine.dc[0].id
  location           = azurerm_resource_group.main.location
  enabled            = true
  
  daily_recurrence_time = "2200"
  timezone              = "Eastern Standard Time"
  
  notification_settings {
    enabled = false
  }
}

resource "azurerm_dev_test_global_vm_shutdown_schedule" "endpoints" {
  count              = var.enable_mini_corp_network ? var.windows_endpoint_count : 0
  virtual_machine_id = azurerm_windows_virtual_machine.endpoints[count.index].id
  location           = azurerm_resource_group.main.location
  enabled            = true
  
  daily_recurrence_time = "2200"
  timezone              = "Eastern Standard Time"
  
  notification_settings {
    enabled = false
  }
}

resource "azurerm_dev_test_global_vm_shutdown_schedule" "servers" {
  count              = var.enable_mini_corp_network ? var.linux_server_count : 0
  virtual_machine_id = azurerm_linux_virtual_machine.servers[count.index].id
  location           = azurerm_resource_group.main.location
  enabled            = true
  
  daily_recurrence_time = "2200"
  timezone              = "Eastern Standard Time"
  
  notification_settings {
    enabled = false
  }
}

