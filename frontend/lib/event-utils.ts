
export const getEventPriority = (event: any) => {
    const eventid = (event.eventid || '').toLowerCase();
    const message = (event.message || '').toLowerCase();

    // Critical: Successful compromise (100)
    if (eventid.includes('login.success')) return 100;
    if (eventid.includes('file_upload') || message.includes('exfiltration')) return 98;
    if (eventid.includes('file_download')) return 95;
    if (message.includes('malware') || message.includes('backdoor') || message.includes('miner')) return 93;

    // High: Post-exploitation commands (70-89)
    if (eventid.includes('command.input') || eventid.includes('command')) {
        if (message.includes('wget') || message.includes('curl -o')) return 88;
        if (message.includes('chmod +x')) return 86;
        if (message.includes('shadow') || message.includes('passwd')) return 85;
        if (message.includes('cron') || message.includes('rc.local') || message.includes('persistence')) return 84;
        if (message.includes('ssh') || message.includes('id_rsa') || message.includes('known_hosts')) return 82;
        if (message.includes('nc ') || message.includes('netcat') || message.includes('4444')) return 87;
        if (message.includes('tar ') || message.includes('base64')) return 83;
        return 75; // Other commands
    }

    // Medium: Authentication attempts (40-60)
    if (eventid.includes('login.failed')) return 45;
    if (eventid.includes('auth') && message.includes('fail')) return 42;

    // Low: Network/Reconnaissance (10-30)
    if (eventid.includes('session.connect')) return 20;
    if (eventid.includes('session.closed')) return 15;
    if (eventid.includes('flow') || eventid.includes('connection')) return 18;

    return 30;
};

export const getSeverityLevel = (priority: number) => {
    if (priority >= 90) return 'CRITICAL';
    if (priority >= 70) return 'HIGH';
    if (priority >= 40) return 'MEDIUM';
    return 'LOW';
};

export const getSeverityColor = (priority: number) => {
     if (priority >= 90) return "text-red-600 dark:text-red-400";
     if (priority >= 70) return "text-orange-600 dark:text-orange-400";
     if (priority >= 40) return "text-yellow-600 dark:text-yellow-400";
     return "text-muted-foreground";
};

export const getSeverityBg = (priority: number) => {
     if (priority >= 90) return "bg-red-100 dark:bg-red-950/40 border-red-300 dark:border-red-700";
     if (priority >= 70) return "bg-orange-100 dark:bg-orange-950/40 border-orange-300 dark:border-orange-700";
     if (priority >= 40) return "bg-yellow-100 dark:bg-yellow-950/40 border-yellow-300 dark:border-yellow-700";
     return "bg-background/50 border-border/50";
};
