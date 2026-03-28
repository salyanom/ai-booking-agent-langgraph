import { motion } from 'framer-motion';
import { useEffect, useState } from 'react';
import { FileText, Download, CheckCircle2, Clock, AlertTriangle, Mail, Pencil } from 'lucide-react';
import { BookingItem, BookingUpdateError, getSummary, SmtpConfig, sendEmailReport, sendMeetingEmailReport, updateBooking } from '@/lib/agentApi';

export const BookingSummary = ({ open, onClose }: { open: boolean; onClose: () => void }) => {
  const [bookings, setBookings] = useState<BookingItem[]>([]);
  const [reportEmail, setReportEmail] = useState('');
  const [reportSubject, setReportSubject] = useState('');
  const [smtpFrom, setSmtpFrom] = useState('');
  const [useCustomSmtp, setUseCustomSmtp] = useState(false);
  const [smtpHost, setSmtpHost] = useState('');
  const [smtpPort, setSmtpPort] = useState('587');
  const [smtpUser, setSmtpUser] = useState('');
  const [smtpPassword, setSmtpPassword] = useState('');
  const [smtpUseTls, setSmtpUseTls] = useState(true);
  const [emailing, setEmailing] = useState(false);
  const [reportMessage, setReportMessage] = useState<string | null>(null);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editingBooking, setEditingBooking] = useState<BookingItem | null>(null);
  const [editTitle, setEditTitle] = useState('');
  const [editDate, setEditDate] = useState('');
  const [editTime, setEditTime] = useState('');
  const [editDuration, setEditDuration] = useState('');
  const [editSuggestions, setEditSuggestions] = useState<string[]>([]);
  const [meetingEmailingId, setMeetingEmailingId] = useState<string | null>(null);

  const loadSummary = async () => {
    try {
      const data = await getSummary();
      setBookings(data.bookings);
    } catch {
      setBookings([]);
    }
  };

  useEffect(() => {
    if (!open) {
      return;
    }

    setReportMessage(null);
    void loadSummary();
  }, [open]);

  const handleEmailReport = async () => {
    const toEmail = reportEmail.trim();
    if (!toEmail) {
      setReportMessage('Enter a recipient email to send the report.');
      return;
    }

    setEmailing(true);
    setReportMessage(null);
    try {
      const smtpConfig: SmtpConfig = {
        host: useCustomSmtp ? smtpHost.trim() : '',
        port: Number(smtpPort || '587'),
        user: useCustomSmtp ? (smtpUser.trim() || undefined) : undefined,
        password: useCustomSmtp ? (smtpPassword || undefined) : undefined,
        from_email: smtpFrom.trim(),
        use_tls: smtpUseTls,
      };

      if (!smtpConfig.from_email) {
        setReportMessage('Sender email is required.');
        return;
      }
      if (useCustomSmtp && !smtpConfig.host) {
        setReportMessage('SMTP host is required when custom SMTP is enabled.');
        return;
      }

      const result = await sendEmailReport(
        toEmail.trim(),
        reportSubject.trim() || undefined,
        smtpConfig,
        undefined,
        'smtp',
      );
      setReportMessage(result.message);
    } catch (error) {
      const text = error instanceof Error ? error.message : 'Failed to send email report.';
      setReportMessage(text);
    } finally {
      setEmailing(false);
    }
  };

  const openEdit = (booking: BookingItem) => {
    setEditingBooking(booking);
    setEditTitle(booking.title);
    setEditDate(booking.date);
    const parsedTime = normalizeTimeInput(booking.time);
    setEditTime(/^\d{2}:\d{2}$/.test(parsedTime) ? parsedTime : '');
    setEditDuration(booking.duration);
    setEditSuggestions([]);
  };

  const closeEdit = () => {
    setEditingBooking(null);
    setEditTitle('');
    setEditDate('');
    setEditTime('');
    setEditDuration('');
    setEditSuggestions([]);
  };

  const normalizeTimeInput = (value: string): string => {
    const trimmed = value.trim();
    const twelveHour = /^(\d{1,2}):(\d{2})\s*(AM|PM)$/i.exec(trimmed);
    if (twelveHour) {
      let hour = parseInt(twelveHour[1], 10);
      const minute = twelveHour[2];
      const ampm = twelveHour[3].toUpperCase();
      if (ampm === 'PM' && hour < 12) hour += 12;
      if (ampm === 'AM' && hour === 12) hour = 0;
      return `${String(hour).padStart(2, '0')}:${minute}`;
    }
    return trimmed;
  };

  const handleEditBookingSave = async () => {
    if (!editingBooking) {
      return;
    }

    setEditingId(editingBooking.id);
    setReportMessage(null);
    setEditSuggestions([]);
    try {
      const result = await updateBooking(editingBooking.id, {
        title: editTitle.trim(),
        date: editDate.trim(),
        time: normalizeTimeInput(editTime),
        duration: editDuration.trim(),
      });
      setReportMessage(result.message);
      await loadSummary();
      window.dispatchEvent(new Event('booking-data-updated'));
      closeEdit();
    } catch (error) {
      if (error instanceof BookingUpdateError) {
        setReportMessage(error.message);
        setEditSuggestions(error.suggestions);
      } else {
        const text = error instanceof Error ? error.message : 'Failed to update booking.';
        setReportMessage(text);
      }
    } finally {
      setEditingId(null);
    }
  };

  const applySuggestionToEdit = (windowText: string) => {
    const match = /^(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2}:\d{2})\s+to\s+(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2}:\d{2})$/.exec(windowText.trim());
    if (!match) {
      return;
    }

    const [, startDate, startTime, , endTime] = match;
    const start = new Date(`${startDate}T${startTime}`);
    const end = new Date(`${startDate}T${endTime}`);
    const durationMinutes = Math.max(30, Math.round((end.getTime() - start.getTime()) / 60000));

    setEditDate(startDate);
    setEditTime(startTime.slice(0, 5));
    setEditDuration(`${durationMinutes} min`);
  };

  const handleMeetingReport = async (booking: BookingItem) => {
    const toEmail = reportEmail.trim();
    if (!toEmail) {
      setReportMessage('Enter a recipient email before sending a meeting report.');
      return;
    }

    setMeetingEmailingId(booking.id);
    setReportMessage(null);
    try {
      const smtpConfig: SmtpConfig = {
        host: useCustomSmtp ? smtpHost.trim() : '',
        port: Number(smtpPort || '587'),
        user: useCustomSmtp ? (smtpUser.trim() || undefined) : undefined,
        password: useCustomSmtp ? (smtpPassword || undefined) : undefined,
        from_email: smtpFrom.trim(),
        use_tls: smtpUseTls,
      };

      if (!smtpConfig.from_email) {
        setReportMessage('Sender email is required.');
        return;
      }
      if (useCustomSmtp && !smtpConfig.host) {
        setReportMessage('SMTP host is required when custom SMTP is enabled.');
        return;
      }

      const result = await sendMeetingEmailReport({
        to_email: toEmail,
        title: booking.title,
        date: booking.date,
        time: booking.time,
        duration: booking.duration,
        status: booking.status,
        subject: reportSubject.trim() || undefined,
        provider: 'smtp',
        smtp: smtpConfig,
      });
      setReportMessage(result.message);
    } catch (error) {
      const text = error instanceof Error ? error.message : 'Failed to send meeting report.';
      setReportMessage(text);
    } finally {
      setMeetingEmailingId(null);
    }
  };

  const handleExportCsv = () => {
    const escapeCsv = (value: string) => `"${value.replace(/"/g, '""')}"`;
    const header = ['title', 'date', 'time', 'duration', 'status'];
    const rows = bookings.map((b) => [
      b.title,
      `='${b.date}'`,
      `='${b.time}'`,
      b.duration,
      b.status,
    ]);
    const csv = [header, ...rows]
      .map((row) => row.map((cell) => escapeCsv(String(cell))).join(','))
      .join('\r\n');

    // Include UTF-8 BOM so Excel opens text cleanly and preserves date/time text fields.
    const blob = new Blob(['\ufeff', csv], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'booking-summary.csv';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  if (!open) return null;

  const confirmed = bookings.filter(b => b.status === 'confirmed').length;
  const pending = bookings.filter(b => b.status === 'pending').length;
  const conflicts = bookings.filter(b => b.status === 'conflict').length;

  return (
    <>
      <div className="fixed inset-0 bg-black/30 backdrop-blur-sm z-[90]" onPointerDown={(e) => { e.preventDefault(); e.stopPropagation(); onClose(); }} />
      <motion.div
        initial={{ opacity: 0, scale: 0.95, y: -10 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        exit={{ opacity: 0, scale: 0.95, y: -10 }}
        className="fixed top-[25%] md:top-[22%] left-1/2 -translate-x-1/2 -translate-y-1/2 z-[100] w-[90vw] max-w-2xl max-h-[88vh] overflow-y-auto glass-card rounded-2xl p-6 shadow-2xl"
      >
        <h2 className="text-base font-bold text-foreground flex items-center gap-2 mb-5">
          <FileText className="h-4 w-4 text-primary" /> Booking Summary
        </h2>

        {/* Summary Stats */}
        <div className="grid grid-cols-3 gap-3 mb-5">
          <div className="glass-button rounded-xl p-3 text-center">
            <CheckCircle2 className="h-5 w-5 text-primary mx-auto mb-1" />
            <p className="text-lg font-bold text-foreground">{confirmed}</p>
            <p className="text-[10px] text-muted-foreground">Confirmed</p>
          </div>
          <div className="glass-button rounded-xl p-3 text-center">
            <Clock className="h-5 w-5 text-accent mx-auto mb-1" />
            <p className="text-lg font-bold text-foreground">{pending}</p>
            <p className="text-[10px] text-muted-foreground">Pending</p>
          </div>
          <div className="glass-button rounded-xl p-3 text-center">
            <AlertTriangle className="h-5 w-5 text-destructive mx-auto mb-1" />
            <p className="text-lg font-bold text-foreground">{conflicts}</p>
            <p className="text-[10px] text-muted-foreground">Conflicts</p>
          </div>
        </div>

        {editingBooking && (
          <div className="glass-button rounded-xl p-3 mb-4">
            <p className="text-xs font-semibold text-foreground mb-2">Edit Booking</p>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-2 mb-2">
              <input
                value={editTitle}
                onChange={(event) => setEditTitle(event.target.value)}
                placeholder="Title"
                className="glass-button rounded-lg px-3 py-2 text-xs text-foreground placeholder:text-muted-foreground"
              />
              <input
                type="date"
                value={editDate}
                onChange={(event) => setEditDate(event.target.value)}
                className="glass-button rounded-lg px-3 py-2 text-xs text-foreground placeholder:text-muted-foreground"
              />
              <input
                type="time"
                value={editTime}
                onChange={(event) => setEditTime(event.target.value)}
                className="glass-button rounded-lg px-3 py-2 text-xs text-foreground placeholder:text-muted-foreground"
              />
              <input
                value={editDuration}
                onChange={(event) => setEditDuration(event.target.value)}
                placeholder="30 min / 1 hr"
                className="glass-button rounded-lg px-3 py-2 text-xs text-foreground placeholder:text-muted-foreground"
              />
            </div>
            {editSuggestions.length > 0 && (
              <div className="mb-2">
                <p className="text-[11px] text-muted-foreground mb-1">Suggested alternatives:</p>
                <div className="flex flex-wrap gap-1.5">
                  {editSuggestions.map((suggestion) => (
                    <button
                      key={suggestion}
                      onClick={() => applySuggestionToEdit(suggestion)}
                      className="glass-button rounded-full px-2.5 py-1 text-[10px] text-primary"
                    >
                      {suggestion}
                    </button>
                  ))}
                </div>
              </div>
            )}
            <div className="flex justify-end gap-2">
              <button
                onClick={closeEdit}
                className="glass-button rounded-lg px-3 py-1.5 text-[11px] text-muted-foreground hover:text-foreground"
              >
                Cancel
              </button>
              <button
                onClick={() => void handleEditBookingSave()}
                disabled={editingId === editingBooking.id}
                className="glass-button rounded-lg px-3 py-1.5 text-[11px] text-primary font-semibold disabled:opacity-50"
              >
                {editingId === editingBooking.id ? 'Saving...' : 'Save Changes'}
              </button>
            </div>
          </div>
        )}

        {/* Booking Table */}
        <div className="glass-button rounded-xl overflow-hidden mb-5">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-border/30">
                <th className="text-left px-3 py-2 text-muted-foreground font-medium">Meeting</th>
                <th className="text-left px-3 py-2 text-muted-foreground font-medium">Date</th>
                <th className="text-left px-3 py-2 text-muted-foreground font-medium">Time</th>
                <th className="text-left px-3 py-2 text-muted-foreground font-medium">Status</th>
                <th className="text-left px-3 py-2 text-muted-foreground font-medium">Actions</th>
              </tr>
            </thead>
          </table>
          <div className="max-h-64 overflow-y-auto">
            <table className="w-full text-xs">
              <tbody>
                {bookings.map(b => {
                  const StatusIcon = b.status === 'confirmed' ? CheckCircle2 : b.status === 'pending' ? Clock : AlertTriangle;
                  const statusColor = b.status === 'confirmed' ? 'text-primary' : b.status === 'pending' ? 'text-accent' : 'text-destructive';
                  return (
                    <tr key={b.id} className="border-b border-border/10 last:border-0">
                      <td className="px-3 py-2 text-foreground font-medium">{b.title}</td>
                      <td className="px-3 py-2 text-muted-foreground">{b.date}</td>
                      <td className="px-3 py-2 text-muted-foreground">{b.time}</td>
                      <td className="px-3 py-2">
                        <StatusIcon className={`h-3.5 w-3.5 ${statusColor}`} />
                      </td>
                      <td className="px-3 py-2">
                        <div className="flex items-center gap-1">
                          {b.status === 'confirmed' ? (
                            <button
                              onClick={() => openEdit(b)}
                              disabled={editingId === b.id}
                              className="glass-button rounded-lg px-2 py-1 text-[10px] text-primary disabled:opacity-50 inline-flex items-center gap-1"
                            >
                              <Pencil className="h-3 w-3" /> Edit
                            </button>
                          ) : (
                            <span className="text-[10px] text-muted-foreground">Read only</span>
                          )}
                          <button
                            onClick={() => void handleMeetingReport(b)}
                            disabled={meetingEmailingId === b.id}
                            className="glass-button rounded-lg px-2 py-1 text-[10px] text-muted-foreground hover:text-foreground inline-flex items-center gap-1 disabled:opacity-50"
                          >
                            <Mail className="h-3 w-3" /> {meetingEmailingId === b.id ? 'Sending...' : 'Report'}
                          </button>
                        </div>
                      </td>
                    </tr>
                  );
                })}
                {bookings.length === 0 && (
                  <tr>
                    <td className="px-3 py-3 text-muted-foreground" colSpan={5}>No bookings yet.</td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>

        {reportMessage && (
          <p className="text-[11px] text-muted-foreground mb-3">{reportMessage}</p>
        )}

        <p className="text-[10px] text-muted-foreground mb-3">SMTP mode enabled for sending reports.</p>

        {/* Actions */}
        <div className="flex flex-wrap justify-end gap-2">
          <div className="flex flex-col sm:flex-row gap-2">
            <input
              type="email"
              value={reportEmail}
              onChange={(event) => setReportEmail(event.target.value)}
              placeholder="recipient@example.com"
              className="glass-button rounded-xl px-3 py-2 text-xs text-foreground placeholder:text-muted-foreground min-w-[220px]"
            />
            <input
              type="email"
              value={smtpFrom}
              onChange={(event) => setSmtpFrom(event.target.value)}
              placeholder="sender@example.com"
              className="glass-button rounded-xl px-3 py-2 text-xs text-foreground placeholder:text-muted-foreground min-w-[220px]"
            />
            <input
              value={reportSubject}
              onChange={(event) => setReportSubject(event.target.value)}
              placeholder="Subject (optional)"
              className="glass-button rounded-xl px-3 py-2 text-xs text-foreground placeholder:text-muted-foreground min-w-[220px]"
            />
          </div>
          <button
            onClick={() => setUseCustomSmtp((value) => !value)}
            className="glass-button rounded-xl px-3 py-2 text-xs text-muted-foreground hover:text-foreground"
          >
            {useCustomSmtp ? 'Hide SMTP' : 'Use Custom SMTP'}
          </button>
          <button
            onClick={() => void handleEmailReport()}
            disabled={emailing || !reportEmail.trim()}
            className="glass-button rounded-xl px-4 py-2 text-xs text-muted-foreground hover:text-foreground flex items-center gap-1.5 disabled:opacity-50"
          >
            <Mail className="h-3.5 w-3.5" /> {emailing ? 'Sending...' : 'Email Report'}
          </button>
          <button
            onClick={handleExportCsv}
            className="glass-button rounded-xl px-4 py-2 text-xs text-primary font-semibold ring-1 ring-primary/30 flex items-center gap-1.5"
          >
            <Download className="h-3.5 w-3.5" /> Export CSV
          </button>
          <button onClick={onClose} className="glass-button rounded-xl px-4 py-2 text-xs text-muted-foreground hover:text-foreground">
            Close
          </button>
        </div>

        {useCustomSmtp && (
          <div className="glass-button rounded-xl p-3 mt-3">
            <p className="text-xs font-semibold text-foreground mb-2">Custom SMTP Settings</p>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
              <input
                value={smtpHost}
                onChange={(event) => setSmtpHost(event.target.value)}
                placeholder="SMTP host (e.g. smtp.gmail.com)"
                className="glass-button rounded-lg px-3 py-2 text-xs text-foreground placeholder:text-muted-foreground"
              />
              <input
                value={smtpPort}
                onChange={(event) => setSmtpPort(event.target.value)}
                placeholder="SMTP port (e.g. 587)"
                className="glass-button rounded-lg px-3 py-2 text-xs text-foreground placeholder:text-muted-foreground"
              />
              <input
                value={smtpUser}
                onChange={(event) => setSmtpUser(event.target.value)}
                placeholder="SMTP user (optional)"
                className="glass-button rounded-lg px-3 py-2 text-xs text-foreground placeholder:text-muted-foreground"
              />
              <input
                type="password"
                value={smtpPassword}
                onChange={(event) => setSmtpPassword(event.target.value)}
                placeholder="SMTP password (optional)"
                className="glass-button rounded-lg px-3 py-2 text-xs text-foreground placeholder:text-muted-foreground"
              />
              <label className="glass-button rounded-lg px-3 py-2 text-xs text-foreground flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={smtpUseTls}
                  onChange={(event) => setSmtpUseTls(event.target.checked)}
                />
                Use TLS
              </label>
            </div>
          </div>
        )}
      </motion.div>
    </>
  );
};
