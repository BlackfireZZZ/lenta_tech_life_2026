export const APP_LAYOUT_COPY = {
  brand: "Lenta · Распознавание ценников",
  nav: [
    { to: "/", label: "Загрузить видео", end: true },
    { to: "/jobs", label: "Мои обработки", end: false },
    { to: "/pipeline", label: "Как это работает", end: false },
    { to: "/experiments", label: "Что мы пробовали", end: false },
    { to: "/shelf", label: "Аудит полки", end: false },
  ],
} as const;

export const ERROR_BOUNDARY_COPY = {
  title: "Что-то пошло не так",
  fallbackMessage: "Непредвиденная ошибка интерфейса.",
  backHome: "На главную",
} as const;
