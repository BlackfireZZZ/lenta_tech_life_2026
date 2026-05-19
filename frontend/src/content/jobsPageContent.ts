import type { BadgeProps } from "@/components/ui/badge";
import type { JobStatus } from "@/api/jobs";

export const JOBS_PAGE_COPY = {
  title: "Мои обработки",
  subtitle: "Все загруженные видео — можно вернуться к любому.",
  uploadButton: "Загрузить видео",
  loadFailed: "Не удалось загрузить список. Проверьте, что бэкенд запущен.",
  emptyTitle: "Пока нет задач",
  emptyHint:
    "Загрузите видео полки — распознанные ценники появятся здесь, и вы всегда сможете к ним вернуться.",
  totalPrefix: "Всего:",
  columns: {
    video: "Видео",
    status: "Статус",
    tags: "Ценников",
    uploaded: "Загружено",
  },
  status: {
    succeeded: { label: "Готово", variant: "success" },
    running: { label: "Обработка", variant: "accent" },
    queued: { label: "В очереди", variant: "neutral" },
    failed: { label: "Ошибка", variant: "danger" },
  } as Record<JobStatus, { label: string; variant: NonNullable<BadgeProps["variant"]> }>,
} as const;
