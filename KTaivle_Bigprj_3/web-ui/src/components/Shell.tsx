"use client";

import { useState } from "react";
import { usePathname } from "next/navigation";
import Link from "next/link";
import Image from "next/image";
import {
    AppShell,
    Burger,
    Group,
    NavLink,
    Text,
    Box,
    Divider,
    Badge,
} from "@mantine/core";
import { useDisclosure } from "@mantine/hooks";
import {
    IconHome,
    IconHeartRateMonitor,
    IconFlask,
    IconFileText,
    IconSettings,
} from "@tabler/icons-react";

const navItems = [
    { href: "/", label: "Dashboard", icon: IconHome },
    { href: "/mpbpk", label: "mPBPK Simulator", icon: IconHeartRateMonitor },
    { href: "/qsar", label: "QSAR Predictor", icon: IconFlask },
    { href: "/reports", label: "Reports", icon: IconFileText },
];

export default function Shell({ children }: { children: React.ReactNode }) {
    const [opened, { toggle }] = useDisclosure();
    const pathname = usePathname();

    return (
        <AppShell
            padding="lg"
            header={{ height: 70 }}
            navbar={{
                width: 280,
                breakpoint: "sm",
                collapsed: { mobile: !opened },
            }}
            styles={{
                main: {
                    backgroundColor: "#F8F9FA",
                    minHeight: "100vh",
                },
            }}
        >
            {/* Header */}
            <AppShell.Header
                style={{
                    backgroundColor: "#FFFFFF",
                    borderBottom: "1px solid #E9ECEF",
                }}
            >
                <Group h="100%" px="md" justify="space-between">
                    <Group>
                        <Burger
                            opened={opened}
                            onClick={toggle}
                            hiddenFrom="sm"
                            size="sm"
                        />
                        <Image
                            src="/certara-logo.png"
                            alt="Certara"
                            width={40}
                            height={40}
                        />
                        <Box>
                            <Text
                                size="lg"
                                fw={700}
                                style={{ color: "#212529", lineHeight: 1.2 }}
                            >
                                SimuPharma™
                            </Text>
                            <Text size="xs" c="dimmed">
                                Clinical Trial Simulator
                            </Text>
                        </Box>
                    </Group>

                    <Group>
                        <Badge color="green" variant="light" size="sm">
                            v1.0
                        </Badge>
                        <Image
                            src="/hanmi-logo.png"
                            alt="Hanmi Pharmaceutical"
                            width={80}
                            height={40}
                            style={{ objectFit: "contain" }}
                        />
                    </Group>
                </Group>
            </AppShell.Header>

            {/* Navbar */}
            <AppShell.Navbar
                p="md"
                style={{ backgroundColor: "#FFFFFF", borderRight: "1px solid #E9ECEF" }}
            >
                <AppShell.Section grow>
                    <Text size="xs" c="dimmed" fw={500} mb="sm" tt="uppercase">
                        Navigation
                    </Text>
                    {navItems.map((item) => (
                        <NavLink
                            key={item.href}
                            component={Link}
                            href={item.href}
                            label={item.label}
                            leftSection={<item.icon size={20} stroke={1.5} />}
                            active={pathname === item.href}
                            variant="light"
                            style={{ borderRadius: 8, marginBottom: 4 }}
                        />
                    ))}
                </AppShell.Section>

                <Divider my="sm" />

                <AppShell.Section>
                    <NavLink
                        href="#"
                        label="Settings"
                        leftSection={<IconSettings size={20} stroke={1.5} />}
                        variant="subtle"
                        style={{ borderRadius: 8 }}
                    />
                    <Box mt="md" p="sm" style={{ backgroundColor: "#F8F9FA", borderRadius: 8 }}>
                        <Text size="xs" c="dimmed">
                            Powered by
                        </Text>
                        <Text size="sm" fw={500}>
                            Certara Technology
                        </Text>
                    </Box>
                </AppShell.Section>
            </AppShell.Navbar>

            {/* Main Content */}
            <AppShell.Main>{children}</AppShell.Main>
        </AppShell>
    );
}
