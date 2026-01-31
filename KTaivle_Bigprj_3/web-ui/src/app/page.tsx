"use client";

import Shell from "@/components/Shell";
import {
  Container,
  Title,
  Text,
  SimpleGrid,
  Card,
  Group,
  ThemeIcon,
  Button,
  Stack,
  Table,
  Badge,
} from "@mantine/core";
import {
  IconActivity,
  IconFlask,
  IconChartBar,
  IconPlayerPlay,
} from "@tabler/icons-react";
import Link from "next/link";

const stats = [
  {
    title: "Total Simulations",
    value: "1,247",
    icon: IconActivity,
    color: "blue",
  },
  {
    title: "Drugs Tested",
    value: "156",
    icon: IconFlask,
    color: "green",
  },
  {
    title: "Success Rate",
    value: "78.3%",
    icon: IconChartBar,
    color: "orange",
  },
];

const recentActivity = [
  { name: "Adalimumab", type: "mPBPK", result: "PASS", date: "10 min ago" },
  { name: "Aspirin", type: "QSAR", result: "LOW RISK", date: "25 min ago" },
  { name: "Pembrolizumab", type: "mPBPK", result: "PASS", date: "1 hour ago" },
  { name: "Ibuprofen", type: "QSAR", result: "MEDIUM RISK", date: "2 hours ago" },
];

export default function Dashboard() {
  return (
    <Shell>
      <Container size="xl">
        {/* Header */}
        <Stack gap="xs" mb="xl">
          <Title order={2} style={{ color: "#212529" }}>
            Dashboard
          </Title>
          <Text c="dimmed" size="sm">
            AI-Driven In-silico Clinical Trial & Toxicity Prediction Platform
          </Text>
        </Stack>

        {/* Stats Cards */}
        <SimpleGrid cols={{ base: 1, sm: 3 }} mb="xl">
          {stats.map((stat) => (
            <Card key={stat.title} shadow="sm" padding="lg" radius="md" withBorder>
              <Group justify="space-between">
                <div>
                  <Text size="xs" c="dimmed" tt="uppercase" fw={500}>
                    {stat.title}
                  </Text>
                  <Text size="xl" fw={700} mt={4}>
                    {stat.value}
                  </Text>
                </div>
                <ThemeIcon size={48} radius="md" variant="light" color={stat.color}>
                  <stat.icon size={24} />
                </ThemeIcon>
              </Group>
            </Card>
          ))}
        </SimpleGrid>

        {/* Quick Actions */}
        <Card shadow="sm" padding="lg" radius="md" withBorder mb="xl">
          <Text fw={600} mb="md">
            Quick Actions
          </Text>
          <Group>
            <Button
              component={Link}
              href="/mpbpk"
              leftSection={<IconPlayerPlay size={16} />}
              variant="filled"
            >
              Run mPBPK Simulation
            </Button>
            <Button
              component={Link}
              href="/qsar"
              leftSection={<IconFlask size={16} />}
              variant="light"
            >
              QSAR Prediction
            </Button>
          </Group>
        </Card>

        {/* Recent Activity */}
        <Card shadow="sm" padding="lg" radius="md" withBorder>
          <Text fw={600} mb="md">
            Recent Activity
          </Text>
          <Table>
            <Table.Thead>
              <Table.Tr>
                <Table.Th>Drug Name</Table.Th>
                <Table.Th>Type</Table.Th>
                <Table.Th>Result</Table.Th>
                <Table.Th>Time</Table.Th>
              </Table.Tr>
            </Table.Thead>
            <Table.Tbody>
              {recentActivity.map((item, idx) => (
                <Table.Tr key={idx}>
                  <Table.Td fw={500}>{item.name}</Table.Td>
                  <Table.Td>
                    <Badge
                      color={item.type === "mPBPK" ? "blue" : "grape"}
                      variant="light"
                      size="sm"
                    >
                      {item.type}
                    </Badge>
                  </Table.Td>
                  <Table.Td>
                    <Badge
                      color={
                        item.result === "PASS" || item.result === "LOW RISK"
                          ? "green"
                          : "yellow"
                      }
                      variant="light"
                      size="sm"
                    >
                      {item.result}
                    </Badge>
                  </Table.Td>
                  <Table.Td c="dimmed">{item.date}</Table.Td>
                </Table.Tr>
              ))}
            </Table.Tbody>
          </Table>
        </Card>
      </Container>
    </Shell>
  );
}
