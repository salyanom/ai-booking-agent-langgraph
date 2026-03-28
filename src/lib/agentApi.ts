export interface AgentChatRequest {
  message: string;
  threadId?: string;
}

export type BookingStatus = 'confirmed' | 'pending' | 'conflict';

export interface BookingItem {
  id: string;
  title: string;
  date: string;
  time: string;
  duration: string;
  participants: string[];
  status: BookingStatus;
}

export interface StatsResponse {
  total_bookings: number;
  confirmed: number;
  pending: number;
  conflicts: number;
}

export interface SummaryResponse {
  stats: StatsResponse;
  bookings: BookingItem[];
}

export interface NotificationItem {
  id: string;
  type: 'success' | 'warning' | 'info';
  title: string;
  message: string;
  time: string;
  read: boolean;
}

export interface NotificationListResponse {
  notifications: NotificationItem[];
  unread_count: number;
}

export interface ImportCsvResponse {
  imported: number;
  skipped: number;
  errors: string[];
}

export interface ActivityItem {
  id: number;
  event_type: string;
  title: string;
  detail: string;
  status: string;
  created_at: string;
}

export interface CalendarEventItem {
  title: string;
  start_time: string;
  end_time: string;
  status: 'confirmed' | 'conflict';
  source?: 'local';
}

export interface EmailReportResponse {
  ok: boolean;
  message: string;
}

export interface SmtpConfig {
  host: string;
  port: number;
  user?: string;
  password?: string;
  from_email: string;
  use_tls: boolean;
}

export interface ResendConfig {
  api_key: string;
  from_email: string;
}

export interface MeetingEmailReportRequest {
  to_email: string;
  title: string;
  date: string;
  time: string;
  duration: string;
  status: string;
  subject?: string;
  provider?: 'smtp' | 'resend';
  smtp?: SmtpConfig;
  resend?: ResendConfig;
}

export interface UpdateBookingRequest {
  title?: string;
  date?: string;
  time?: string;
  duration?: string;
}

export interface UpdateBookingResponse {
  ok: boolean;
  booking: BookingItem;
  message: string;
}

export class BookingUpdateError extends Error {
  suggestions: string[];

  constructor(message: string, suggestions: string[] = []) {
    super(message);
    this.name = 'BookingUpdateError';
    this.suggestions = suggestions;
  }
}

export interface AgentChatResponse {
  reply: string;
  thread_id: string;
  booking_status: string;
  conflict_suggestions: string[];
  action_options: string[];
  state: {
    current_intent?: Record<string, unknown>;
  };
}

const API_BASE_URL = import.meta.env.VITE_AGENT_API_URL ?? 'http://127.0.0.1:8000';

async function parseResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let detailText = '';
    try {
      const body = await response.json();
      const detail = body?.detail;
      if (typeof detail === 'string') {
        detailText = detail;
      } else if (detail && typeof detail === 'object') {
        const message = typeof detail.message === 'string' ? detail.message : JSON.stringify(detail);
        const suggestions = Array.isArray(detail.suggestions) ? detail.suggestions : [];
        if (suggestions.length > 0) {
          detailText = `${message} Suggested alternatives: ${suggestions.join(' | ')}`;
        } else {
          detailText = message;
        }
      } else {
        detailText = JSON.stringify(body);
      }
    } catch {
      detailText = await response.text();
    }
    throw new Error(`Agent API error ${response.status}: ${detailText}`);
  }

  return response.json() as Promise<T>;
}

export async function sendAgentMessage(
  payload: AgentChatRequest,
): Promise<AgentChatResponse> {
  const response = await fetch(`${API_BASE_URL}/api/chat`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      message: payload.message,
      thread_id: payload.threadId,
    }),
  });

  return parseResponse<AgentChatResponse>(response);
}

export async function getBookings(): Promise<BookingItem[]> {
  const response = await fetch(`${API_BASE_URL}/api/bookings`);
  return parseResponse<BookingItem[]>(response);
}

export async function getStats(): Promise<StatsResponse> {
  const response = await fetch(`${API_BASE_URL}/api/stats`);
  return parseResponse<StatsResponse>(response);
}

export async function getSummary(): Promise<SummaryResponse> {
  const response = await fetch(`${API_BASE_URL}/api/summary`);
  return parseResponse<SummaryResponse>(response);
}

export async function getNotifications(): Promise<NotificationListResponse> {
  const response = await fetch(`${API_BASE_URL}/api/notifications`);
  return parseResponse<NotificationListResponse>(response);
}

export async function markAllNotificationsRead(): Promise<number> {
  const response = await fetch(`${API_BASE_URL}/api/notifications/mark-all-read`, {
    method: 'POST',
  });
  const data = await parseResponse<{ unread_count: number }>(response);
  return data.unread_count;
}

export async function dismissNotification(notificationId: string): Promise<number> {
  const response = await fetch(`${API_BASE_URL}/api/notifications/${notificationId}`, {
    method: 'DELETE',
  });
  const data = await parseResponse<{ unread_count: number }>(response);
  return data.unread_count;
}

export async function importBookingsCsv(csvContent: string): Promise<ImportCsvResponse> {
  const response = await fetch(`${API_BASE_URL}/api/bookings/import-csv`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ csv_content: csvContent }),
  });

  return parseResponse<ImportCsvResponse>(response);
}

export async function getActivity(): Promise<ActivityItem[]> {
  const response = await fetch(`${API_BASE_URL}/api/activity`);
  return parseResponse<ActivityItem[]>(response);
}

function formatDateTimeForApi(value: Date): string {
  const pad = (n: number) => String(n).padStart(2, '0');
  return `${value.getFullYear()}-${pad(value.getMonth() + 1)}-${pad(value.getDate())} ${pad(value.getHours())}:${pad(value.getMinutes())}:${pad(value.getSeconds())}`;
}

export async function getCalendarEvents(start: Date, end: Date): Promise<CalendarEventItem[]> {
  const query = new URLSearchParams({
    start: formatDateTimeForApi(start),
    end: formatDateTimeForApi(end),
  });
  const response = await fetch(`${API_BASE_URL}/api/calendar/events?${query.toString()}`);
  return parseResponse<CalendarEventItem[]>(response);
}

export async function updateBooking(
  bookingId: string,
  payload: UpdateBookingRequest,
): Promise<UpdateBookingResponse> {
  const response = await fetch(`${API_BASE_URL}/api/bookings/${bookingId}`, {
    method: 'PATCH',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    try {
      const body = await response.json();
      const detail = body?.detail;
      if (detail && typeof detail === 'object') {
        const message = typeof detail.message === 'string' ? detail.message : `Agent API error ${response.status}`;
        const suggestions = Array.isArray(detail.suggestions)
          ? detail.suggestions.filter((item: unknown): item is string => typeof item === 'string')
          : [];
        throw new BookingUpdateError(message, suggestions);
      }
      if (typeof detail === 'string') {
        throw new BookingUpdateError(detail, []);
      }
      throw new BookingUpdateError(`Agent API error ${response.status}: ${JSON.stringify(body)}`, []);
    } catch (error) {
      if (error instanceof BookingUpdateError) {
        throw error;
      }
      const text = await response.text();
      throw new BookingUpdateError(`Agent API error ${response.status}: ${text}`, []);
    }
  }

  return response.json() as Promise<UpdateBookingResponse>;
}

export async function sendEmailReport(
  toEmail: string,
  subject?: string,
  smtp?: SmtpConfig,
  resend?: ResendConfig,
  provider: 'smtp' | 'resend' = 'smtp',
): Promise<EmailReportResponse> {
  const response = await fetch(`${API_BASE_URL}/api/reports/email`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      to_email: toEmail,
      subject,
      provider,
      smtp,
      resend,
    }),
  });
  return parseResponse<EmailReportResponse>(response);
}

export async function sendMeetingEmailReport(payload: MeetingEmailReportRequest): Promise<EmailReportResponse> {
  const response = await fetch(`${API_BASE_URL}/api/reports/meeting-email`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  });
  return parseResponse<EmailReportResponse>(response);
}
