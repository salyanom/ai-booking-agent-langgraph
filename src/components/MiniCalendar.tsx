import { useEffect, useMemo, useState } from 'react';
import { motion } from 'framer-motion';
import { ChevronLeft, ChevronRight } from 'lucide-react';
import {
  CalendarEventItem,
  getCalendarEvents,
} from '@/lib/agentApi';

const DAYS = ['Su', 'Mo', 'Tu', 'We', 'Th', 'Fr', 'Sa'];
const MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'];

type ViewMode = 'month' | 'week' | 'day';

function startOfDay(value: Date): Date {
  return new Date(value.getFullYear(), value.getMonth(), value.getDate(), 0, 0, 0);
}

function addDays(value: Date, days: number): Date {
  const next = new Date(value);
  next.setDate(next.getDate() + days);
  return next;
}

function formatDayKey(value: Date): string {
  const y = value.getFullYear();
  const m = String(value.getMonth() + 1).padStart(2, '0');
  const d = String(value.getDate()).padStart(2, '0');
  return `${y}-${m}-${d}`;
}

function parseEventDate(value: string): Date {
  return new Date(value.replace(' ', 'T'));
}

function formatTime(value: Date): string {
  return value.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

export const MiniCalendar = () => {
  const [view, setView] = useState<ViewMode>('month');
  const [focusDate, setFocusDate] = useState(startOfDay(new Date()));
  const [events, setEvents] = useState<CalendarEventItem[]>([]);

  const periodRange = useMemo(() => {
    const dayStart = startOfDay(focusDate);

    if (view === 'day') {
      return { start: dayStart, end: addDays(dayStart, 1) };
    }

    if (view === 'week') {
      const offset = dayStart.getDay();
      const weekStart = addDays(dayStart, -offset);
      return { start: weekStart, end: addDays(weekStart, 7) };
    }

    const monthStart = new Date(dayStart.getFullYear(), dayStart.getMonth(), 1, 0, 0, 0);
    const nextMonth = new Date(dayStart.getFullYear(), dayStart.getMonth() + 1, 1, 0, 0, 0);
    return { start: monthStart, end: nextMonth };
  }, [focusDate, view]);

  useEffect(() => {
    const load = async () => {
      try {
        const data = await getCalendarEvents(periodRange.start, periodRange.end);
        setEvents(data);
      } catch {
        setEvents([]);
      }
    };

    void load();
  }, [periodRange.start, periodRange.end]);

  const eventsByDay = useMemo(() => {
    const map = new Map<string, CalendarEventItem[]>();
    for (const item of events) {
      const key = formatDayKey(parseEventDate(item.start_time));
      const list = map.get(key) ?? [];
      list.push(item);
      map.set(key, list);
    }
    return map;
  }, [events]);

  const year = focusDate.getFullYear();
  const month = focusDate.getMonth();
  const firstDay = new Date(year, month, 1).getDay();
  const daysInMonth = new Date(year, month + 1, 0).getDate();
  const todayKey = formatDayKey(new Date());

  const cells: (number | null)[] = [];
  for (let i = 0; i < firstDay; i++) {
    cells.push(null);
  }
  for (let d = 1; d <= daysInMonth; d++) {
    cells.push(d);
  }

  const weekDays = useMemo(() => {
    const offset = focusDate.getDay();
    const start = addDays(startOfDay(focusDate), -offset);
    return Array.from({ length: 7 }, (_, i) => addDays(start, i));
  }, [focusDate]);

  const dayEvents = useMemo(() => {
    const key = formatDayKey(focusDate);
    return eventsByDay.get(key) ?? [];
  }, [eventsByDay, focusDate]);

  const navigate = (direction: -1 | 1) => {
    if (view === 'day') {
      setFocusDate(prev => addDays(prev, direction));
      return;
    }
    if (view === 'week') {
      setFocusDate(prev => addDays(prev, direction * 7));
      return;
    }
    setFocusDate(prev => new Date(prev.getFullYear(), prev.getMonth() + direction, 1));
  };

  const viewLabel =
    view === 'day'
      ? focusDate.toLocaleDateString([], { month: 'short', day: 'numeric', year: 'numeric' })
      : `${MONTHS[month]} ${year}`;

  return (
    <div className="glass-card rounded-2xl p-4 w-full">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-xs font-semibold text-foreground">
          {viewLabel}
        </h3>
        <div className="flex gap-1">
          <button onClick={() => navigate(-1)} className="glass-button rounded-lg p-1"><ChevronLeft className="h-3 w-3 text-muted-foreground" /></button>
          <button onClick={() => navigate(1)} className="glass-button rounded-lg p-1"><ChevronRight className="h-3 w-3 text-muted-foreground" /></button>
        </div>
      </div>

      <div className="flex items-center gap-1 mb-3">
        {(['month', 'week', 'day'] as ViewMode[]).map(mode => (
          <button
            key={mode}
            onClick={() => setView(mode)}
            className={`glass-button rounded-full px-2.5 py-1 text-[10px] font-medium ${
              view === mode ? 'ring-1 ring-primary/40 text-primary' : 'text-muted-foreground hover:text-foreground'
            }`}
          >
            {mode[0].toUpperCase() + mode.slice(1)}
          </button>
        ))}
      </div>

      {view === 'month' && (
        <>
          <div className="grid grid-cols-7 gap-0.5">
            {DAYS.map(d => (
              <div key={d} className="text-[10px] text-muted-foreground text-center py-1 font-medium">{d}</div>
            ))}
            {cells.map((day, i) => {
              const dayDate = day ? new Date(year, month, day) : null;
              const key = dayDate ? formatDayKey(dayDate) : '';
              const dayItems = key ? eventsByDay.get(key) ?? [] : [];
              const isToday = key === todayKey;
              const hasConflict = dayItems.some(item => item.status === 'conflict');
              const hasBooking = dayItems.length > 0;

              return (
                <motion.button
                  key={i}
                  whileHover={{ scale: 1.12 }}
                  whileTap={{ scale: 0.92 }}
                  onClick={() => {
                    if (dayDate) {
                      setFocusDate(startOfDay(dayDate));
                    }
                  }}
                  className={`relative text-[11px] rounded-lg py-1.5 transition-colors
                    ${!day ? 'invisible' : 'hover:bg-primary/10 cursor-pointer'}
                    ${isToday ? 'bg-primary text-primary-foreground font-bold' : 'text-foreground'}
                  `}
                >
                  {day}
                  {hasBooking && !isToday && (
                    <span className={`absolute bottom-0.5 left-1/2 -translate-x-1/2 w-1 h-1 rounded-full ${hasConflict ? 'bg-destructive' : 'bg-primary'}`} />
                  )}
                </motion.button>
              );
            })}
          </div>

          <div className="flex items-center gap-4 mt-3 pt-2 border-t border-border/30">
            <div className="flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-primary" />
              <span className="text-[10px] text-muted-foreground">Booked</span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-destructive" />
              <span className="text-[10px] text-muted-foreground">Conflict</span>
            </div>
          </div>
        </>
      )}

      {view === 'week' && (
        <div className="space-y-2 max-h-56 overflow-y-auto pr-1">
          {weekDays.map(day => {
            const key = formatDayKey(day);
            const items = eventsByDay.get(key) ?? [];
            return (
              <div key={key} className="glass-button rounded-xl p-2.5">
                <div className="flex items-center justify-between mb-1">
                  <p className="text-[11px] font-medium text-foreground">
                    {day.toLocaleDateString([], { weekday: 'short', month: 'short', day: 'numeric' })}
                  </p>
                  <span className="text-[10px] text-muted-foreground">
                    {items.filter(item => item.status === 'confirmed').length} booked / {items.filter(item => item.status === 'conflict').length} conflict
                  </span>
                </div>
                {items.length === 0 && <p className="text-[10px] text-muted-foreground">No events</p>}
                {items.map((item, index) => (
                  <p key={`${item.title}-${index}`} className={`text-[10px] ${item.status === 'conflict' ? 'text-destructive' : 'text-muted-foreground'}`}>
                    {formatTime(parseEventDate(item.start_time))} - {item.title}
                  </p>
                ))}
              </div>
            );
          })}
        </div>
      )}

      {view === 'day' && (
        <div className="space-y-2 max-h-56 overflow-y-auto pr-1">
          <p className="text-[11px] font-medium text-foreground">
            {focusDate.toLocaleDateString([], { weekday: 'long', month: 'long', day: 'numeric' })}
          </p>
          <p className="text-[10px] text-muted-foreground">
            {dayEvents.filter(item => item.status === 'confirmed').length} booked / {dayEvents.filter(item => item.status === 'conflict').length} conflict
          </p>
          {dayEvents.length === 0 && <p className="text-[10px] text-muted-foreground">No events scheduled.</p>}
          {dayEvents.map((item, index) => (
            <div key={`${item.title}-${index}`} className="glass-button rounded-xl p-2.5">
              <p className="text-[11px] font-medium text-foreground truncate">{item.title}</p>
              <p className={`text-[10px] ${item.status === 'conflict' ? 'text-destructive' : 'text-muted-foreground'}`}>
                {formatTime(parseEventDate(item.start_time))} - {formatTime(parseEventDate(item.end_time))}
              </p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};
