import { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { CheckCircle2, RefreshCw, AlertTriangle, Clock, Ban } from 'lucide-react';
import { ActivityItem, getActivity } from '@/lib/agentApi';

function toRelativeTime(value: string): string {
  const createdAt = new Date(value.replace(' ', 'T'));
  if (Number.isNaN(createdAt.getTime())) {
    return value;
  }

  const diffMs = Date.now() - createdAt.getTime();
  const mins = Math.max(1, Math.floor(diffMs / 60000));
  if (mins < 60) {
    return `${mins} min ago`;
  }

  const hours = Math.floor(mins / 60);
  if (hours < 24) {
    return `${hours} hr ago`;
  }

  const days = Math.floor(hours / 24);
  return `${days} day ago`;
}

function activityVisual(activity: ActivityItem): {
  icon: typeof Clock;
  color: string;
} {
  const event = activity.event_type.toLowerCase();
  const status = activity.status.toLowerCase();

  if (event.includes('reschedule') || status === 'rescheduled') {
    return { icon: RefreshCw, color: 'text-accent' };
  }
  if (event.includes('cancel') || status === 'cancelled') {
    return { icon: Ban, color: 'text-muted-foreground' };
  }
  if (event.includes('conflict') || status === 'conflict') {
    return { icon: AlertTriangle, color: 'text-destructive' };
  }
  if (event.includes('booking') || status === 'confirmed') {
    return { icon: CheckCircle2, color: 'text-primary' };
  }
  return { icon: Clock, color: 'text-muted-foreground' };
}

export const ActivityFeed = () => {
  const [activities, setActivities] = useState<ActivityItem[]>([]);

  useEffect(() => {
    const load = async () => {
      try {
        const data = await getActivity();
        setActivities(data);
      } catch {
        setActivities([]);
      }
    };

    void load();
    const handler = () => {
      void load();
    };

    window.addEventListener('booking-data-updated', handler);
    return () => window.removeEventListener('booking-data-updated', handler);
  }, []);

  return (
    <div className="space-y-2 pb-4">
      <h3 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-3 flex items-center gap-2">
        <Clock className="h-3.5 w-3.5" /> Recent Activity
      </h3>
      {activities.map((a, i) => {
        const visual = activityVisual(a);
        const Icon = visual.icon;

        return (
        <motion.div
          key={a.id}
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: i * 0.06 }}
          className="glass-button rounded-xl p-2.5 flex items-center gap-3"
        >
          <div className={`shrink-0 ${visual.color}`}>
            <Icon className="h-3.5 w-3.5" />
          </div>
          <div className="flex-1 min-w-0">
            <p className="text-[11px] font-medium text-foreground truncate">{a.title}</p>
            <p className="text-[10px] text-muted-foreground truncate">{a.detail}</p>
          </div>
          <span className="text-[9px] text-muted-foreground/60 shrink-0">{toRelativeTime(a.created_at)}</span>
        </motion.div>
        );
      })}
      {activities.length === 0 && (
        <p className="text-xs text-muted-foreground text-center py-3">No activity yet.</p>
      )}
    </div>
  );
};
