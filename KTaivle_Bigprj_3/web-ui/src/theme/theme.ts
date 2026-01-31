import { createTheme, MantineColorsTuple } from "@mantine/core";

// Certara Orange palette
const certaraOrange: MantineColorsTuple = [
    "#fff4e6",
    "#ffe8cc",
    "#ffd8a8",
    "#ffc078",
    "#ffa94d",
    "#F7941D", // Primary - Certara Orange
    "#e68a1a",
    "#cc7a17",
    "#b36b14",
    "#995c11",
];

export const theme = createTheme({
    primaryColor: "certara",
    colors: {
        certara: certaraOrange,
    },
    fontFamily:
        "Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif",
    headings: {
        fontFamily:
            "Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif",
        fontWeight: "600",
    },
    defaultRadius: "md",
    primaryShade: 5,
    white: "#FFFFFF",
    black: "#212529",
    other: {
        backgroundLight: "#F8F9FA",
        textSecondary: "#868E96",
    },
});
