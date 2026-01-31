import Shell from "@/components/Shell";
import {
    Container,
    Title,
    Text,
    Card,
    Stack,
    Table,
    Badge,
    Button,
    Group,
} from "@mantine/core";
import { IconDownload, IconFileText } from "@tabler/icons-react";

const reports = [
    {
        id: 1,
        name: "mPBPK Validation Report",
        type: "PDF",
        date: "2026-01-27",
        status: "Complete",
    },
    {
        id: 2,
        name: "QSAR ClinTox Validation",
        type: "PDF",
        date: "2026-01-27",
        status: "Complete",
    },
    {
        id: 3,
        name: "Cohort Simulation - EUR/EAS",
        type: "CSV",
        date: "2026-01-26",
        status: "Complete",
    },
    {
        id: 4,
        name: "Adalimumab PK Profile",
        type: "PDF",
        date: "2026-01-25",
        status: "Complete",
    },
];

export default function ReportsPage() {
    return (
        <Shell>
            <Container size="xl">
                <Stack gap="xs" mb="xl">
                    <Title order={2}>Reports</Title>
                    <Text c="dimmed" size="sm">
                        Download simulation results and validation reports
                    </Text>
                </Stack>

                <Card shadow="sm" padding="lg" radius="md" withBorder>
                    <Group justify="space-between" mb="md">
                        <Text fw={600}>Available Reports</Text>
                        <Button
                            variant="light"
                            leftSection={<IconFileText size={16} />}
                            size="sm"
                        >
                            Generate New Report
                        </Button>
                    </Group>

                    <Table>
                        <Table.Thead>
                            <Table.Tr>
                                <Table.Th>Report Name</Table.Th>
                                <Table.Th>Type</Table.Th>
                                <Table.Th>Date</Table.Th>
                                <Table.Th>Status</Table.Th>
                                <Table.Th>Action</Table.Th>
                            </Table.Tr>
                        </Table.Thead>
                        <Table.Tbody>
                            {reports.map((report) => (
                                <Table.Tr key={report.id}>
                                    <Table.Td fw={500}>{report.name}</Table.Td>
                                    <Table.Td>
                                        <Badge color="blue" variant="light" size="sm">
                                            {report.type}
                                        </Badge>
                                    </Table.Td>
                                    <Table.Td c="dimmed">{report.date}</Table.Td>
                                    <Table.Td>
                                        <Badge color="green" variant="light" size="sm">
                                            {report.status}
                                        </Badge>
                                    </Table.Td>
                                    <Table.Td>
                                        <Button
                                            variant="subtle"
                                            size="xs"
                                            leftSection={<IconDownload size={14} />}
                                        >
                                            Download
                                        </Button>
                                    </Table.Td>
                                </Table.Tr>
                            ))}
                        </Table.Tbody>
                    </Table>
                </Card>
            </Container>
        </Shell>
    );
}
