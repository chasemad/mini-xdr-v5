"use client";

import React, { useMemo } from 'react';
import { BarChart, Bar, XAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';

interface TimelineEvent {
  ts: string;
  eventid: string;
  [key: string]: any;
}

interface InteractiveTimelineProps {
  events: TimelineEvent[];
}

export default function InteractiveTimeline({ events }: InteractiveTimelineProps) {
  const data = useMemo(() => {
    if (!events || events.length === 0) return [];

    // Sort events by time
    const sortedEvents = [...events].sort((a, b) => new Date(a.ts).getTime() - new Date(b.ts).getTime());

    if (sortedEvents.length === 0) return [];

    const startTime = new Date(sortedEvents[0].ts).getTime();
    const endTime = new Date(sortedEvents[sortedEvents.length - 1].ts).getTime();
    const duration = endTime - startTime;

    // Determine bucket size
    let bucketSizeMS = 1000 * 60; // 1 minute default
    if (duration > 1000 * 60 * 60 * 24) {
        bucketSizeMS = 1000 * 60 * 60; // 1 hour
    } else if (duration > 1000 * 60 * 60) {
        bucketSizeMS = 1000 * 60 * 15; // 15 mins
    }

    const buckets = new Map<string, number>();
    const timeFormat = new Intl.DateTimeFormat('en-US', {
        hour: 'numeric',
        minute: 'numeric',
        hour12: false
    });

    sortedEvents.forEach(e => {
        const time = new Date(e.ts).getTime();
        const bucketTime = Math.floor(time / bucketSizeMS) * bucketSizeMS;
        const key = timeFormat.format(new Date(bucketTime));
        buckets.set(key, (buckets.get(key) || 0) + 1);
    });

    return Array.from(buckets.entries()).map(([time, count]) => ({
        time,
        count
    }));

  }, [events]);

  if (data.length === 0) {
    return (
        <div className="h-full w-full flex items-center justify-center text-muted-foreground text-sm">
            No timeline data available
        </div>
    );
  }

  return (
    <div className="w-full h-full min-h-[100px] bg-transparent">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data}>
          <XAxis
            dataKey="time"
            stroke="#888888"
            fontSize={10}
            tickLine={false}
            axisLine={false}
            minTickGap={20}
          />
          <Tooltip
            contentStyle={{
                backgroundColor: 'hsl(var(--popover))',
                borderColor: 'hsl(var(--border))',
                color: 'hsl(var(--popover-foreground))',
                fontSize: '12px',
                borderRadius: '6px'
            }}
            itemStyle={{ color: 'hsl(var(--foreground))' }}
            cursor={{ fill: 'hsl(var(--muted)/0.3)' }}
          />
          <Bar dataKey="count" fill="hsl(var(--primary))" radius={[2, 2, 0, 0]}>
            {data.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={`hsl(var(--primary) / ${0.6 + (entry.count / Math.max(...data.map(d => d.count)) * 0.4)})`} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
